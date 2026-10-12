import time
import numpy as np
import cv2
from pathlib import Path
from ops import gray_loop, gray_numpy, gray_cv2

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

for name, fn in [("loop", gray_loop), ("numpy", gray_numpy), ("cv2", gray_cv2)]:
    med, p95 = bench(fn, img)
    print(f"gray {name:6} median {med:9.3f} ms   p95 {p95:9.3f} ms")