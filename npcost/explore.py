import numpy as np, time

# img = np.zeros((480, 640, 3), np.uint8)
# print("strides:", img.strides)
# print("C-contiguous:", img.flags["C_CONTIGUOUS"])
# print("nbytes:", img.nbytes)

# chw = img.transpose(2, 0, 1)
# print("chw shape:", chw.shape, "strides:", chw.strides)
# print("chw contiguous:", chw.flags["C_CONTIGUOUS"])
# print("chw is a view of img:", chw.base is img)

# chw2 = np.ascontiguousarray(chw)
# print("chw2 strides:", chw2.strides, "contiguous:", chw2.flags["C_CONTIGUOUS"])
# print("chw2 shares memory with img:", np.shares_memory(chw2, img))

# sl = img[::2, ::2]
# print("slice shape:", sl.shape, "strides:", sl.strides)
# print("slice contiguous:", sl.flags["C_CONTIGUOUS"], "view:", sl.base is img)

a = np.uint8(200); b = np.uint8(100)
print("uint8 add:", a + b)                       # what do you expect?
arr = np.array([200], np.uint8)
print("array add:", arr + np.uint8(100))
print("safe:", arr.astype(np.float32) + 100)

for dt in (np.uint8, np.float16, np.float32):
    x = np.zeros((480, 640, 3), dt)
    print(dt.__name__, x.nbytes / 1e6, "MB")