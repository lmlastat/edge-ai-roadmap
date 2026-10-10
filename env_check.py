#!/usr/bin/env python3
"""Edge AI Ascent - Day 1 environment check.

Run inside your edge venv:
    python env_check.py

Checks the RTX 5070 (Blackwell, sm_120) setup: PyTorch 2.7+ built with
CUDA 12.8+, a working GPU matmul, and the packages the roadmap needs.
Exit code 0 = all required checks passed, 1 = at least one failed.
"""
import importlib
import platform
import shutil
import subprocess
import sys

REQUIRED_TORCH = (2, 7)
REQUIRED_CUDA = (12, 8)

results = []  # (status, name, detail)


def record(status, name, detail=""):
    results.append((status, name, detail))
    mark = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[INFO]"}[status]
    print(f"{mark} {name}" + (f": {detail}" if detail else ""))


def parse_version(text):
    parts = []
    for chunk in str(text).split("+")[0].split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_python():
    record("INFO", "Python", f"{platform.python_version()} ({sys.executable})")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        record("PASS", "Virtual environment active", sys.prefix)
    else:
        record("WARN", "Virtual environment active", "not in a venv - activate your edge venv first")


def check_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        record("FAIL", "nvidia-smi", "not found - NVIDIA driver not installed or not on PATH")
        return
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        if out.returncode == 0 and line:
            record("PASS", "NVIDIA driver", line)
        else:
            record("FAIL", "NVIDIA driver", out.stderr.strip() or "no GPU reported")
    except Exception as exc:
        record("FAIL", "nvidia-smi", repr(exc))


def check_torch():
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        record("FAIL", "import torch", repr(exc))
        return

    tv = parse_version(torch.__version__)
    if tv[:2] >= REQUIRED_TORCH:
        record("PASS", "PyTorch version", f"{torch.__version__} (need >= 2.7)")
    else:
        record("FAIL", "PyTorch version", f"{torch.__version__} - need >= 2.7 for Blackwell")

    cuda_built = torch.version.cuda
    if not cuda_built:
        record("FAIL", "PyTorch CUDA build", "CPU-only build installed")
        return
    cv = parse_version(cuda_built)
    if cv[:2] >= REQUIRED_CUDA:
        record("PASS", "PyTorch CUDA build", f"{cuda_built} (need >= 12.8)")
    else:
        record("FAIL", "PyTorch CUDA build", f"{cuda_built} - need >= 12.8 for Blackwell")

    if not torch.cuda.is_available():
        record("FAIL", "torch.cuda.is_available()", "False")
        return
    record("PASS", "torch.cuda.is_available()", "True")

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    record("INFO", "GPU", f"{name}, compute capability {major}.{minor}, {total_gb:.1f} GB")

    arch_list = torch.cuda.get_arch_list()
    record("INFO", "Compiled arch list", " ".join(arch_list))
    sm = f"sm_{major}{minor}"
    if sm in arch_list or any(a.startswith("compute_") and a.endswith(f"{major}{minor}") for a in arch_list):
        record("PASS", f"Kernels include {sm}", "")
    else:
        record("FAIL", f"Kernels include {sm}", "this build has no kernels for your GPU")

    try:
        for dtype in (torch.float32, torch.float16):
            a = torch.randn(1024, 1024, device="cuda", dtype=dtype)
            b = torch.randn(1024, 1024, device="cuda", dtype=dtype)
            c = a @ b
            torch.cuda.synchronize()
            ok = bool(torch.isfinite(c).all().item())
            record("PASS" if ok else "FAIL", f"GPU matmul {str(dtype).split('.')[-1]}", "finite result" if ok else "NaN/inf")
    except Exception as exc:
        record("FAIL", "GPU matmul", repr(exc))

    bench_matmul(torch)


def bench_matmul(torch, n=4096, warmup=20, iters=100):
    """Time an n x n matmul in FP32 and FP16 (Day 1 deliverable)."""
    timings = {}
    try:
        for dtype in (torch.float32, torch.float16):
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            for _ in range(warmup):          # settle clocks and autotuning
                a @ b
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                a @ b
                end.record()
                torch.cuda.synchronize()      # GPU work is async: wait before reading
                times.append(start.elapsed_time(end))
            times.sort()
            median = times[len(times) // 2]
            p95 = times[int(len(times) * 0.95) - 1]
            tflops = 2 * n ** 3 / (median / 1e3) / 1e12
            timings[dtype] = median
            name = str(dtype).split(".")[-1]
            record("INFO", f"{n}x{n} matmul {name}",
                   f"median {median:.3f} ms, p95 {p95:.3f} ms, {tflops:.1f} TFLOPS")
        speedup = timings[torch.float32] / timings[torch.float16]
        record("INFO", "FP16 speedup over FP32", f"{speedup:.2f}x")
    except Exception as exc:
        record("FAIL", f"{n}x{n} matmul timing", repr(exc))


def check_packages():
    # (import name, pip name, required?)
    packages = [
        ("torchvision", "torchvision", True),
        ("numpy", "numpy", True),
        ("cv2", "opencv-python", False),
        ("onnx", "onnx", False),
        ("onnxruntime", "onnxruntime-gpu", False),
        ("tensorrt", "tensorrt", False),
        ("ultralytics", "ultralytics", False),
    ]
    for module, pip_name, required in packages:
        try:
            mod = importlib.import_module(module)
            record("PASS", module, getattr(mod, "__version__", "installed"))
        except Exception:
            record("FAIL" if required else "WARN", module, f"missing - pip install {pip_name}")

    try:
        ort = importlib.import_module("onnxruntime")
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            record("PASS", "onnxruntime CUDA provider", "available")
        else:
            record("WARN", "onnxruntime CUDA provider", f"not available, providers: {providers}")
    except Exception:
        pass


def main():
    print("Edge AI Ascent - environment check\n" + "-" * 40)
    check_python()
    check_nvidia_smi()
    check_torch()
    check_packages()
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print("-" * 40)
    print(f"{len(fails)} failed, {len(warns)} warnings")
    if fails:
        print("Fix the FAIL items above, then re-run.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
