import numpy as np
import cv2

def gray_loop(img: np.ndarray) -> np.ndarray:
    # img is HxWx3 uint8 in BGR order (that's how cv2 loads images)
    h, w, _ = img.shape
    out = np.zeros((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            b, g, r = img[y, x]
            out[y, x] = 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)
    return out

def gray_cv2(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gray_numpy(img: np.ndarray) -> np.ndarray:
    # img is HxWx3 uint8 in BGR order (that's how cv2 loads images)
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    out = 0.299 * r + 0.587 * g + 0.114 * b
    return out.astype(np.uint8)

img = cv2.imread("rainforest.jpg")

# gray = gray_loop(img)
# cv2.imwrite("rainforest_gray_loop.jpg", gray)

# gray = gray_cv2(img)
# cv2.imwrite("rainforest_gray_cv2.jpg", gray)

# gray = gray_numpy(img)
# cv2.imwrite("rainforest_gray_numpy.jpg", gray)


a = gray_loop(img); b = gray_numpy(img); c = gray_cv2(img)
print("loop vs cv2:", np.abs(a.astype(int) - c.astype(int)).max())
print("np vs cv2:  ", np.abs(b.astype(int) - c.astype(int)).max())