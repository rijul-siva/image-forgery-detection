"""
Integration tests for the full PixelGuard end-to-end pipeline.
Verified and executed by Member C.
"""

import json
import os
import tempfile
import cv2
import numpy as np
import pytest

from app.streamlit_app import run_full_pipeline, report_to_dict, safe_upload_suffix
from src.utils.data_types import ForensicReport


def test_safe_upload_suffix_preserves_format():
    """The temp-file suffix must follow the uploaded container format."""
    assert safe_upload_suffix("photo.png") == ".png"
    assert safe_upload_suffix("photo.JPG") == ".jpg"
    assert safe_upload_suffix("photo.jpeg") == ".jpeg"
    # Unknown / missing extensions fall back to JPEG.
    assert safe_upload_suffix("photo") == ".jpg"
    assert safe_upload_suffix("photo.gif") == ".jpg"
    assert safe_upload_suffix("") == ".jpg"


@pytest.fixture
def genuine_test_image():
    """Generates an authentic synthetic photo with distinct background and no copy-paste."""
    rng = np.random.default_rng(101)
    img = np.full((600, 800, 3), 180, dtype=np.uint8)
    # Add random distinct features
    for _ in range(50):
        cx = int(rng.integers(50, 750))
        cy = int(rng.integers(50, 550))
        r = int(rng.integers(5, 20))
        col = (int(rng.integers(0, 255)), int(rng.integers(0, 255)), int(rng.integers(0, 255)))
        cv2.circle(img, (cx, cy), r, col, -1)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def forged_test_image():
    """Generates a synthetic forged image with an object duplicated at a translated position."""
    img = np.full((600, 800, 3), 140, dtype=np.uint8)
    rng = np.random.default_rng(202)

    # Distinct source object patch
    patch = np.full((100, 100, 3), 230, dtype=np.uint8)
    for _ in range(30):
        cx = int(rng.integers(10, 90))
        cy = int(rng.integers(10, 90))
        r = int(rng.integers(3, 8))
        cv2.circle(patch, (cx, cy), r, (int(rng.integers(20, 240)), 40, int(rng.integers(20, 240))), -1)
    cv2.rectangle(patch, (15, 15), (45, 80), (10, 10, 200), -1)

    # Paste source at (120, 120)
    img[120:220, 120:220] = patch
    # Paste clone at (450, 300)
    img[300:400, 450:550] = patch

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_pipeline_on_genuine_image(genuine_test_image):
    """Pipeline on genuine image should not detect forgery."""
    report = run_full_pipeline(
        genuine_test_image,
        detector="orb",
        ratio=0.75,
        ransac_threshold=3.0
    )
    assert isinstance(report, ForensicReport)
    assert report.preprocess.original_width == 800
    assert report.preprocess.original_height == 600
    assert report.copy_move.forgery_detected is False
    assert report.ela is not None
    assert report.metadata is not None


def test_pipeline_on_forged_image(forged_test_image):
    """Pipeline on copy-move image should detect the duplicated region."""
    report = run_full_pipeline(
        forged_test_image,
        detector="orb",
        ratio=0.85,
        ransac_threshold=5.0
    )
    assert isinstance(report, ForensicReport)
    assert report.copy_move.forgery_detected is True
    assert report.copy_move.num_inlier_matches >= 4
    assert report.copy_move.transform_params is not None
    tp = report.copy_move.transform_params
    assert tp.scale_factor == pytest.approx(1.0, abs=0.1)


def test_report_serialization_and_json(forged_test_image):
    """Verify ForensicReport serialization to dict and JSON."""
    report = run_full_pipeline(forged_test_image, detector="orb")
    report_dict = report_to_dict(report)
    assert isinstance(report_dict, dict)
    assert "preprocessing" in report_dict
    assert "copy_move_detection" in report_dict
    assert "ela_analysis" in report_dict
    assert "metadata_analysis" in report_dict

    # Check JSON serializability without errors
    json_str = json.dumps(report_dict, indent=2)
    assert len(json_str) > 0
    loaded = json.loads(json_str)
    assert loaded["preprocessing"]["original_width"] == 800
