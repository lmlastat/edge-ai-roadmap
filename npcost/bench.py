import time
import numpy as np
import cv2
from pathlib import Path
from ops import box_blur_loop, box_blur_cv2, box_blur_np, gray_loop, gray_numpy, gray_cv2, resize_cv2, resize_loop, resize_np

def bench(fn, *args, warmup=3, runs=50):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1e3)
    return np.median(times), np.percentile(times, 95)

HERE = Path(__file__).parent
img = cv2.resize(cv2.imread(str(HERE / "rainforest.jpg")), (320, 240))
gray = gray_loop(img)

ops = {
    "grayscale": ((gray_loop, gray_numpy, gray_cv2), img),
    "3x3 blur":  ((box_blur_loop, box_blur_np, box_blur_cv2), gray),
    "resize":    ((resize_loop, resize_np, resize_cv2), gray),
}

lines = [
    "| op | loop (ms) | numpy (ms) | cv2 (ms) | loop/numpy | numpy/cv2 |",
    "|---|---|---|---|---|---|",
]
for name, (fns, data) in ops.items():
    med = [bench(fn, data)[0] for fn in fns]
    lines.append(f"| {name} | {med[0]:.3f} | {med[1]:.3f} | {med[2]:.3f} | "
                 f"{med[0]/med[1]:.0f}x | {med[1]/med[2]:.1f}x |")

table = "\n".join(lines)
print(table)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "bench.md").write_text(table + "\n")