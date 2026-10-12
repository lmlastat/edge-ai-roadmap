import time
import numpy as np
import cv2
from pathlib import Path
from npcost.ops import box_blur_loop, box_blur_cv2, box_blur_np, gray_loop, gray_numpy, gray_cv2

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

ops = [
    ("gray", [("loop", gray_loop), ("numpy", gray_numpy), ("cv2", gray_cv2)], img),
    ("blur", [("loop", box_blur_loop), ("numpy", box_blur_np), ("cv2", box_blur_cv2)], gray),
]


for op_name, impls, data in ops:
    for impl_name, fn in impls:
        med, p95 = bench(fn, data)
        print(f"{op_name:5} {impl_name:6} median {med:9.3f} ms   p95 {p95:9.3f} ms")