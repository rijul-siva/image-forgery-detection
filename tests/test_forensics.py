"""
Tests for Member B's supporting forensics modules: ELA and Metadata.
Verified and executed by Member C.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest
from PIL import Image

from src.forensics.ela import run_ela_analysis
from src.forensics.metadata import run_metadata_analysis
from src.utils.data_types import ELAResult, MetadataResult


@pytest.fixture
def temp_jpeg_image():
    """Create a temporary standard JPEG image."""
    img = np.full((300, 400, 3), 150, dtype=np.uint8)
    cv2.circle(img, (150, 150), 40, (0, 0, 255), -1)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_png_image():
    """Create a temporary standard PNG image."""
    img = np.full((200, 200, 3), 100, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


# ---- 1. Error Level Analysis (ELA) Tests ----

def test_ela_analysis_valid_jpeg(temp_jpeg_image):
    res = run_ela_analysis(temp_jpeg_image, quality=95, amplification=20)
    assert isinstance(res, ELAResult)
    assert res.ela_image is not None
    assert res.ela_image.ndim == 3
    assert res.ela_image.dtype == np.uint8
    assert res.suspicion_level in ["low", "medium", "high"]
    assert len(res.description) > 0


def test_ela_analysis_png(temp_png_image):
    """PNG recompressed to JPEG should produce a valid ELA result."""
    res = run_ela_analysis(temp_png_image, quality=90, amplification=15)
    assert isinstance(res, ELAResult)
    assert res.ela_image.shape == (200, 200, 3)


def test_ela_analysis_invalid_path():
    with pytest.raises(ValueError, match="Could not load image"):
        run_ela_analysis("non_existent_file_path_xyz.jpg")


# ---- 2. Metadata / EXIF Analysis Tests ----

def test_metadata_stripped(temp_jpeg_image):
    """Synthesized cv2 image has no EXIF tags; should report metadata_stripped=True."""
    res = run_metadata_analysis(temp_jpeg_image)
    assert isinstance(res, MetadataResult)
    assert res.metadata_stripped is True
    assert res.has_editing_software is False
    assert res.editing_software is None
    assert any("missing or stripped" in flag for flag in res.flags)


def test_metadata_with_camera_info():
    """Create an image containing simulated Camera Make and Model EXIF tags via Pillow."""
    img = Image.new("RGB", (100, 100), color="white")
    exif = img.getexif()
    # 0x010F = Make, 0x0110 = Model
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPhone 14 Pro"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, exif=exif)
        path = tmp.name

    try:
        res = run_metadata_analysis(path)
        assert res.metadata_stripped is False
        assert "Make" in res.metadata_dict
        assert "Model" in res.metadata_dict
        assert "Apple" in str(res.metadata_dict["Make"])
        assert "iPhone 14 Pro" in str(res.metadata_dict["Model"])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_metadata_with_editing_software():
    """Create an image containing Photoshop software signature via Pillow."""
    img = Image.new("RGB", (100, 100), color="blue")
    exif = img.getexif()
    # 0x0131 = Software
    exif[0x0131] = "Adobe Photoshop 24.1 (Windows)"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, exif=exif)
        path = tmp.name

    try:
        res = run_metadata_analysis(path)
        assert res.has_editing_software is True
        assert res.editing_software is not None
        assert "Photoshop" in res.editing_software
        assert any("Editing software tag identified" in f for f in res.flags)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_metadata_invalid_path():
    """Invalid path should be handled gracefully without crashing."""
    res = run_metadata_analysis("totally_non_existent_file.jpg")
    assert isinstance(res, MetadataResult)
    assert res.metadata_stripped is True
    assert len(res.flags) > 0
