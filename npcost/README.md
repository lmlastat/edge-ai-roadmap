# npcost: the cost of a loop

Grayscale, 3×3 box blur and bilinear resize, each written three ways
(pure Python loops, vectorized NumPy, OpenCV) and benchmarked.

## Results (median of 50 runs)
| op | loop (ms) | numpy (ms) | cv2 (ms) | loop/numpy | numpy/cv2 |
|---|---|---|---|---|---|
| grayscale | 45.181 | 0.131 | 0.014 | 346x | 9.1x |
| 3x3 blur | 184.628 | 0.213 | 0.036 | 865x | 5.9x |
| resize | 27.696 | 1.046 | 0.024 | 26x | 44.2x |

## Setup
Python 3.14.4, NumPy 2.5.2, OpenCV 5.0.0, Ryzen 7 7800X3D.
Input 320×240 grayscale/BGR photo; resize outputs 224×224.

## What I observed
1. loop is the slowest, it loops through every pixel and has overhead
2. numpy is second because it loops less. Takes advantage of layers and broadcasting to cut down on operations. 
3. cv2 is fastest beacuse it works entirely in uint8 in a single pass and skips the overhead of floating point

## Reproduce
python bench.py
