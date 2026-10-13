import numpy as np
import pytest

from edgebench_lite import imageops as ops


@pytest.fixture
def bgr():
    rng = np.random.default_rng(0)  # fixed seed: same image every run
    return rng.integers(
        0, 256, size=(24, 32, 3), dtype=np.uint8
    )  # small: the loops are slow


@pytest.fixture
def gray(bgr):
    return ops.gray_cv2(bgr)


def max_diff(a, b):
    # astype(int) first: subtracting uint8 arrays directly would wrap around
    return int(np.abs(a.astype(int) - b.astype(int)).max())


@pytest.mark.parametrize("fn", [ops.gray_loop, ops.gray_numpy])
def test_gray_matches_cv2(fn, bgr):
    assert max_diff(fn(bgr), ops.gray_cv2(bgr)) <= 1


@pytest.mark.parametrize("fn", [ops.box_blur_loop, ops.box_blur_np])
def test_box_blur_matches_cv2(fn, gray):
    assert max_diff(fn(gray), ops.box_blur_cv2(gray)) <= 1


@pytest.mark.parametrize("fn", [ops.resize_loop, ops.resize_np])
def test_resize_matches_cv2(fn, gray):
    out_h, out_w = 16, 20
    assert max_diff(fn(gray, out_h, out_w), ops.resize_cv2(gray, out_h, out_w)) <= 1


@pytest.mark.parametrize("fn", [ops.resize_loop, ops.resize_np, ops.resize_cv2])
def test_output_shape_and_dtype(fn, gray):
    assert gray.shape == (24, 32)
    out_h, out_w = 16, 20
    out = fn(gray, out_h, out_w)
    assert out.shape == (out_h, out_w)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("fn", [ops.resize_loop, ops.resize_np, ops.resize_cv2])
def test_resize_same_size_is_identity(fn, gray):
    out = fn(gray, *gray.shape)
    assert np.array_equal(out, gray)


@pytest.mark.parametrize("fn", [ops.box_blur_loop, ops.box_blur_cv2, ops.box_blur_np])
def test_blur_constant_image(fn, gray):
    # blurring a constant image should return the same constant image
    const = np.full(gray.shape, 100, dtype=np.uint8)
    out = fn(const)
    assert np.array_equal(out, const)


@pytest.mark.parametrize("fn", [ops.gray_loop, ops.gray_cv2, ops.gray_numpy])
def test_tiny_images_do_not_crash(fn, gray):
    # tiny images (1x1, 1x2, 2x1) should not crash any of the functions
    for h in [1, 2]:
        for w in [1, 2]:
            tiny = np.full((h, w), 100, dtype=np.uint8)
            if fn.__name__.startswith("gray"):
                # gray functions expect a 3-channel image
                tiny = np.stack([tiny] * 3, axis=-1)
            out = fn(tiny)
            assert out.shape == (h, w)
            assert out.dtype == np.uint8
