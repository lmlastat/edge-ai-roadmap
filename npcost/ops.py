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

def box_blur_loop(img: np.ndarray) -> np.ndarray:
    # gray is HxW uint8
    h, w = img.shape
    K = np.full((3, 3), 1/9, np.float32)  # 3x3 box blur kernel
    p = np.pad(img, 1, mode="reflect")   #padded input
    out= np.zeros((h, w), np.uint8)

    for y in range(h):
        for x in range(w):
            acc = 0.0
            for dy in range(3):
                for dx in range(3):
                    acc += K[dy, dx] * p[y + dy, x + dx]
            out[y, x] = min(max(int(round(acc)), 0), 255)
    return out

def box_blur_cv2(gray: np.ndarray) -> np.ndarray:
    K = np.full((3, 3), 1 / 9, np.float32)
    return cv2.filter2D(gray, -1, K)      # -1 means "output the same dtype as the input"

def box_blur_np(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    K = np.full((3, 3), 1 / 9, np.float32)
    p = np.pad(gray, 1, mode="reflect").astype(np.float32)   # float, so sums can't overflow
    acc = np.zeros((h, w), np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += K[dy, dx] * p[dy:dy+h, dx:dx+w]
    return np.clip(np.round(acc), 0, 255).astype(np.uint8)

img = cv2.imread("rainforest.jpg")

gray = gray_loop(img)
# cv2.imwrite("results/rainforest_gray_loop.jpg", gray)
# box_blur = box_blur_loop(gray)
# cv2.imwrite("results/rainforest_box_blur_loop.jpg", box_blur)

# gray = gray_cv2(img)
# cv2.imwrite("results/rainforest_gray_cv2.jpg", gray)
# box_blur = box_blur_cv2(gray)
# cv2.imwrite("results/rainforest_box_blur_cv2.jpg", box_blur)

# gray = gray_numpy(img)
# cv2.imwrite("results/rainforest_gray_numpy.jpg", gray)
# box_blur = box_blur_np(gray)
# cv2.imwrite("results/rainforest_box_blur_numpy.jpg", box_blur)


# a = gray_loop(img); b = gray_numpy(img); c = gray_cv2(img)
# print("loop vs cv2:", np.abs(a.astype(int) - c.astype(int)).max())
# print("np vs cv2:  ", np.abs(b.astype(int) - c.astype(int)).max())


a = box_blur_loop(gray); b = box_blur_np(gray); c = box_blur_cv2(gray)
print("loop vs cv2:", np.abs(a.astype(int) - c.astype(int)).max())
print("np vs cv2:  ", np.abs(b.astype(int) - c.astype(int)).max())

