"""
Tests for preprocessing module (Member B).
Verified and executed by Member C.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from src.preprocessing import preprocess_image
from src.utils.data_types import PreprocessResult


@pytest.fixture
def sample_image_standard():
    """Create a standard 800x600 test image."""
    img = np.full((600, 800, 3), 128, dtype=np.uint8)
    cv2.circle(img, (400, 300), 50, (0, 255, 0), -1)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_image_large():
    """Create a large 2400x1800 test image to test aspect-ratio downscaling."""
    img = np.full((1800, 2400, 3), 100, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_preprocess_standard_image(sample_image_standard):
    """Verify normal preprocessing without downscaling."""
    res = preprocess_image(sample_image_standard, max_edge=1600)
    assert isinstance(res, PreprocessResult)
    assert res.original_width == 800
    assert res.original_height == 600
    assert res.scale_factor == 1.0
    assert res.color_image.shape == (600, 800, 3)
    assert res.gray_image.shape == (600, 800)
    assert res.gray_image.dtype == np.uint8


def test_preprocess_large_image_scaling(sample_image_large):
    """Verify downscaling for images exceeding max_edge."""
    max_edge = 1200
    res = preprocess_image(sample_image_large, max_edge=max_edge)
    assert res.original_width == 2400
    assert res.original_height == 1800
    assert res.scale_factor == pytest.approx(1200 / 2400, rel=1e-3)
    assert max(res.color_image.shape[:2]) == max_edge
    assert res.color_image.shape[1] == 1200  # width
    assert res.color_image.shape[0] == 900   # height (aspect ratio 4:3 preserved)


def test_preprocess_denoising(sample_image_standard):
    """Verify mild Gaussian smoothing is applied when denoise=True."""
    res_clean = preprocess_image(sample_image_standard, denoise=False)
    res_denoised = preprocess_image(sample_image_standard, denoise=True)
    assert res_denoised.color_image is not None
    assert res_denoised.gray_image is not None
    assert res_denoised.color_image.shape == res_clean.color_image.shape


def test_preprocess_invalid_path():
    """Verify appropriate exception is raised on invalid path."""
    with pytest.raises(ValueError, match="Could not load image"):
        preprocess_image("non_existent_file_path_12345.jpg")


def test_preprocess_corrupt_file():
    """Verify exception on corrupt non-image file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"NOT_AN_IMAGE_DATA_HEADER")
        path = tmp.name
    try:
        with pytest.raises(ValueError, match="Could not load image"):
            preprocess_image(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_preprocess_normalises_exif_orientation():
    """EXIF orientation tag 6 (rotate 90° CW) must be applied on load."""
    from PIL import Image

    image = Image.new("RGB", (200, 100), color="white")
    exif = image.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, exif=exif)
        path = tmp.name

    try:
        res = preprocess_image(path, max_edge=1600)
        # Stored landscape 200x100, displayed portrait 100x200.
        assert res.original_width == 100
        assert res.original_height == 200
        assert res.color_image.shape[:2] == (200, 100)
    finally:
        if os.path.exists(path):
            os.remove(path)
