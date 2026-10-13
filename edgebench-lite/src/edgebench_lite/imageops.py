import numpy as np
import cv2
import math

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


def resize_loop(gray: np.ndarray, out_h: int = 224, out_w: int = 224) -> np.ndarray:
    in_h, in_w = gray.shape
    scale_y = in_h / out_h
    scale_x = in_w / out_w
    out = np.zeros((out_h, out_w), np.uint8)
    for oy in range(out_h):
        for ox in range(out_w):
            sy = (oy + 0.5) * scale_y - 0.5
            sx = (ox + 0.5) * scale_x - 0.5
            y0 = int(math.floor(sy)); x0 = int(math.floor(sx))
            fy = sy - y0;         fx = sx - x0
            y1 = y0 + 1;          x1 = x0 + 1

            y0 = np.clip(y0, 0, in_h - 1);  y1 = np.clip(y1, 0, in_h - 1)  # clamp all four AFTER computing fy, fx
            x0 = np.clip(x0, 0, in_w - 1);  x1 = np.clip(x1, 0, in_w - 1)
            # clamp y0, y1, x0, x1 into range (step 3)
            A = float(gray[y0, x0]); B = float(gray[y0, x1]); C = float(gray[y1, x0]); D = float(gray[y1, x1])
            top = A * (1 - fx) + B * fx
            bottom = C * (1 - fx) + D * fx
            out[oy, ox] = int(round(top * (1 - fy) + bottom * fy))
    return out

def resize_cv2(gray, out_h=224, out_w=224):
    return cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR)   # (width, height) order

def resize_np(gray, out_h=224, out_w=224):
    in_h, in_w = gray.shape
    scale_y = in_h / out_h
    scale_x = in_w / out_w

    sy = (np.arange(out_h) + 0.5) * scale_y - 0.5     # shape (out_h,)  one per output row
    sx = (np.arange(out_w) + 0.5) * scale_x - 0.5     # shape (out_w,)  one per output column

    y0 = np.floor(sy).astype(int);  fy = sy - y0      # fractional part, before clamping
    x0 = np.floor(sx).astype(int);  fx = sx - x0      # fractional part, before clamping
    y1 = y0 + 1;                    x1 = x0 + 1

    y0 = np.clip(y0, 0, in_h - 1);  y1 = np.clip(y1, 0, in_h - 1)  # clamp all four AFTER computing fy, fx
    x0 = np.clip(x0, 0, in_w - 1);  x1 = np.clip(x1, 0, in_w - 1)

    g = gray.astype(np.float32)
    A = g[y0[:, None], x0[None, :]]                   # shape (out_h, out_w)
    B = g[y0[:, None], x1[None, :]]                   # same, with x1
    C = g[y1[:, None], x0[None, :]]                   # y1, x0
    D = g[y1[:, None], x1[None, :]]                   # y1, x1

    fx = fx[None, :]                                  # shape (1, out_w)  broadcasts across rows
    fy = fy[:, None]                                  # shape (out_h, 1)  broadcasts across columns
    top    = A * (1 - fx) + B * fx
    bottom = C * (1 - fx) + D * fx
    value  = top * (1 - fy) + bottom * fy
    return np.round(np.clip(value, 0, 255)).astype(np.uint8)  # round, clip 0..255, uint8
