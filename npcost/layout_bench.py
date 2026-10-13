import time
import numpy as np

def med_ms(fn, runs=50, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))

rng = np.random.default_rng(0)
hwc = rng.random((480, 640, 3), dtype=np.float32)                # how cv2 stores images
chw = np.ascontiguousarray(hwc.transpose(2, 0, 1))               # what a GPU model wants
chw_view = hwc.transpose(2, 0, 1)                                # same memory as hwc, new strides
f16 = hwc.astype(np.float16)

rows = [
    ("per-channel mean, HWC",              lambda: hwc.mean(axis=(0, 1))),
    ("per-channel mean, CHW (contiguous)", lambda: chw.mean(axis=(1, 2))),
    ("per-channel mean, CHW (view)",       lambda: chw_view.mean(axis=(1, 2))),
    ("HWC -> CHW copy",                    lambda: np.ascontiguousarray(hwc.transpose(2, 0, 1))),
    ("x*0.5+1, float32",                   lambda: hwc * 0.5 + 1),
    ("x*0.5+1, float16",                   lambda: f16 * np.float16(0.5) + np.float16(1)),
]

print("| experiment | median ms |\n|---|---|")
for name, fn in rows:
    print(f"| {name} | {med_ms(fn):.3f} |")