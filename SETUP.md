# Setup

Dev environment for the Edge AI Ascent roadmap. Everything through day 49 runs on the desktop GPU.

## Hardware

| Part | Detail |
|------|--------|
| CPU | Ryzen 7 7800X3D |
| GPU | NVIDIA GeForce RTX 5070, 12 GB (Blackwell, compute capability 12.0 / `sm_120`) |
| RAM | 32 GB DDR5 |
| OS | Ubuntu (add exact version: `lsb_release -d`) |

## Software (as verified by `env_check.py`)

| Component | Version |
|-----------|---------|
| NVIDIA driver | 595.99.02 |
| Python | 3.14.4 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| numpy | 2.5.2 |

Blackwell GPUs need a PyTorch build with CUDA 12.8 or newer (PyTorch 2.7+). Older builds
import fine but have no kernels for `sm_120`, so GPU work fails at runtime.

## Create the environment

```bash
python3 -m venv ~/venvs/edge
source ~/venvs/edge/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Verify

```bash
python env_check.py
```

It checks the driver, the PyTorch and CUDA versions, that the build contains `sm_120`
kernels, FP32 and FP16 matmuls on the GPU, and which of the later packages are
installed. It also times a 4096x4096 matmul in FP32 and FP16.

Exit code 0 means every required check passed. Warnings are packages needed later.

## Packages for later projects (not installed yet)

| Package | Needed for |
|---------|------------|
| opencv-python, ultralytics | P1 Desk Inventory (day 8) |
| onnx, onnxruntime-gpu | P3 Inference ladder (day 22) |
| tensorrt | P3 and P5 |

Python 3.14 is new, so some of these may not have wheels yet. If an install fails,
create a second venv on Python 3.12 or 3.13.

## Day 1 timing results

Paste the output of `python env_check.py` here after running it (matmul FP32 vs FP16).

Edge AI Ascent - environment check
----------------------------------------
[INFO] Python: 3.14.4 (/home/edge-dev/venvs/edge/bin/python)
[ OK ] Virtual environment active: /home/edge-dev/venvs/edge
[ OK ] NVIDIA driver: NVIDIA GeForce RTX 5070, 595.99.02, 12227 MiB
[ OK ] PyTorch version: 2.11.0+cu128 (need >= 2.7)
[ OK ] PyTorch CUDA build: 12.8 (need >= 12.8)
[ OK ] torch.cuda.is_available(): True
[INFO] GPU: NVIDIA GeForce RTX 5070, compute capability 12.0, 11.5 GB
[INFO] Compiled arch list: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
[ OK ] Kernels include sm_120
[ OK ] GPU matmul float32: finite result
[ OK ] GPU matmul float16: finite result
[INFO] 4096x4096 matmul float32: median 6.168 ms, p95 6.190 ms, 22.3 TFLOPS
[INFO] 4096x4096 matmul float16: median 2.031 ms, p95 2.035 ms, 67.7 TFLOPS
[INFO] FP16 speedup over FP32: 3.04x
[ OK ] torchvision: 0.26.0+cu128
[ OK ] numpy: 2.5.2
[WARN] cv2: missing - pip install opencv-python
[WARN] onnx: missing - pip install onnx
[WARN] onnxruntime: missing - pip install onnxruntime-gpu
[WARN] tensorrt: missing - pip install tensorrt
[WARN] ultralytics: missing - pip install ultralytics
----------------------------------------
0 failed, 5 warnings

## `nvidia-smi` notes

One line per column of the default output (see Day 1 exercise in the roadmap).
4096x4096 is the dimensions of the matrix.
matmul is the operation.
float32 and float12 is the precision in terms of bits, 32 bit vs 12 bit.
median is the middle value of the 120-20= 100 timed runs. first 20 are thrown out due to gpu warming up.
p95 takes the 95 runs out of the 100 runs and finds the cutoff point of speed of 95% of requests are guaranteed to be faster than. 
TFLOPS is the unit for how many trillions of floating point operations your gpu can do per second.
