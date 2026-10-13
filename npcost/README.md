# npcost: the cost of a loop

Grayscale, 3×3 box blur and bilinear resize, each written three ways
(pure Python loops, vectorized NumPy, OpenCV) and benchmarked.

## Results (median of 50 runs)
| op | loop (ms) | numpy (ms) | cv2 (ms) | loop/numpy | numpy/cv2 |
|---|---|---|---|---|---|
| grayscale | 45.181 | 0.131 | 0.014 | 346x | 9.1x |
| 3x3 blur | 184.628 | 0.213 | 0.036 | 865x | 5.9x |
| resize | 27.696 | 1.046 | 0.024 | 26x | 44.2x |


## Layout and dtype experiment
Random float32 image, 480×640×3, median of 50 runs.

| experiment | median ms |
|---|---|
| per-channel mean, HWC | 3.699 |
| per-channel mean, CHW (contiguous) | 0.135 |
| per-channel mean, CHW (view) | 3.691 |
| HWC -> CHW copy | 0.212 |
| x*0.5+1, float32 | 2.678 |
| x*0.5+1, float16 | 7.363 |

## Setup
Python 3.14.4, NumPy 2.5.2, OpenCV 5.0.0, Ryzen 7 7800X3D.
Input 320×240 grayscale/BGR photo; resize outputs 224×224.

## What I observed
1. loop is the slowest, it loops through every pixel and has overhead
2. numpy is second because it loops less. Takes advantage of layers and broadcasting to cut down on operations. 
3. cv2 is fastest beacuse it works entirely in uint8 in a single pass and skips the overhead of floating point
4. Cv2 returns numpy array in format HWC, with channel being BGR format. pytorch libraries use a different format, namely CHW and RGB as convention.
5. A view is free to create but not to use. CHW contiguous mean was 27× faster than the view; the copy cost 0.2 ms.
6. View doesn't change memory, it gives you stride which tells you how far you have to step in each axis to get to the next pixel info.
7. Contigous copy (np.ascontigousarray(view)) reorders memory but makes it easy to get channel info in a single pass.
8. Contigous is faster than view because straight-ordered memory surpasses the cost of creating a copy. View is slower because while its free it still requires you to jump through memory.
9. float16 was 2.75× slower than float32 on this CPU because numpy has no native float16 calculation.

## Reproduce
python bench.py            # writes results/bench.md
python layout_bench.py     # layout and dtype table

## Limitations

- Single machine; absolute numbers will differ elsewhere, ratios should hold roughly.
- Loops and NumPy versions are written for clarity, not tuned.
- Resize NumPy version builds temporary arrays via fancy indexing; a tuned version would be faster.

## Interview Hour

“Why is a NumPy operation so much faster than the same loop in Python?”
- A python loop is slow mainly due to overhead from the interpretor which adds significant time. Numpy operations on the other hand use compiled c code to run calculations and has same-typed numbers lined up in contiguous memory blocks avoiding any slow overhead. It also utilizes SIMD instructions which allow multiple values in a single instruction.

"Why does HWC against CHW matter for a model’s input"
- Mostly it’s convention: PyTorch models expect RGB, CHW tensors. Converting from cv2’s HWC means a transpose, which is just a view, followed by a copy to make it contiguous in the new order. In my benchmark that copy cost about 0.2 ms, and a per-channel mean on the contiguous CHW array was 27× faster than on the HWC data, because each channel’s values are stored together.
