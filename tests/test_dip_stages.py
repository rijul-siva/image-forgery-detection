"""
Tests for Individual Digital Image Processing (DIP) Stages.
Verifies that every stage of the DIP pipeline produces verifiable intermediate artifacts
for faculty inspection and academic review.
"""

import numpy as np
import cv2
import pytest
import os
import tempfile

from src.preprocessing import preprocess_image
from src.copy_move.pipeline import run_copy_move_analysis
from src.forensics.ela import run_ela_analysis


@pytest.fixture
def synthetic_forged_image():
    """Create a temporary synthetic image with a known copy-paste region."""
    img = np.full((600, 800, 3), 140, dtype=np.uint8)
    # Background texture
    cv2.rectangle(img, (50, 50), (250, 450), (60, 60, 60), -1)
    
    # Feature-rich patch at source (x=100, y=100)
    for i in range(25):
        cv2.circle(img, (120 + (i % 5) * 20, 150 + (i // 5) * 20), 8, (200, 50, 50), -1)
        cv2.rectangle(img, (110 + i * 5, 200), (120 + i * 5, 240), (40, 200, 40), -1)

    # Copy to destination (x=450, y=100)
    patch = img[100:300, 100:250].copy()
    img[100:300, 450:600] = patch

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_dip_stage_1_preprocessing(synthetic_forged_image):
    """Test DIP Stage 1: Grayscale conversion, spatial decimation, and Gaussian smoothing."""
    prep = preprocess_image(synthetic_forged_image, max_edge=1600, denoise=False)
    
    # Check Grayscale conversion
    assert prep.gray_image is not None
    assert prep.gray_image.ndim == 2
    assert prep.gray_image.shape == (600, 800)
    assert prep.gray_image.dtype == np.uint8

    # Check Gaussian smoothed representation
    assert prep.denoised_image is not None
    assert prep.denoised_image.shape == (600, 800, 3)
    assert prep.denoised_image.dtype == np.uint8


def test_dip_stage_2_feature_extraction(synthetic_forged_image):
    """Test DIP Stage 2: Scale-space keypoint detection visualization."""
    prep = preprocess_image(synthetic_forged_image, max_edge=1600)
    cm = run_copy_move_analysis(prep.gray_image, prep.color_image, detector="orb")
    
    # Keypoint rich visualization
    assert cm.keypoints_visualization is not None
    assert cm.keypoints_visualization.shape == (600, 800, 3)
    assert cm.keypoints_visualization.dtype == np.uint8


def test_dip_stage_3_feature_matching(synthetic_forged_image):
    """Test DIP Stage 3: Candidate match pairs after ratio test and spatial suppression."""
    prep = preprocess_image(synthetic_forged_image, max_edge=1600)
    cm = run_copy_move_analysis(prep.gray_image, prep.color_image, detector="orb")
    
    assert cm.raw_matches_visualization is not None
    assert cm.raw_matches_visualization.shape == (600, 800, 3)
    assert cm.num_total_matches > 0


def test_dip_stage_4_geometric_ransac(synthetic_forged_image):
    """Test DIP Stage 4: RANSAC inlier consensus vs outlier classification."""
    prep = preprocess_image(synthetic_forged_image, max_edge=1600)
    cm = run_copy_move_analysis(prep.gray_image, prep.color_image, detector="orb")
    
    assert cm.match_visualization is not None
    assert cm.match_visualization.shape == (600, 800, 3)
    assert cm.forgery_detected is True
    assert cm.affine_matrix is not None
    assert cm.affine_matrix.shape == (2, 3)
    assert cm.transform_params is not None


def test_dip_stage_5_morphology_and_segmentation(synthetic_forged_image):
    """Test DIP Stage 5: Binary keypoint mask -> Closing -> Dilation -> Region Contours."""
    prep = preprocess_image(synthetic_forged_image, max_edge=1600)
    cm = run_copy_move_analysis(prep.gray_image, prep.color_image, detector="orb")
    
    # Raw binary footprint mask
    assert cm.raw_mask is not None
    assert cm.raw_mask.shape == (600, 800)
    assert cm.raw_mask.dtype == np.uint8

    # Morphological closing (A • B)
    assert cm.closed_mask is not None
    assert cm.closed_mask.shape == (600, 800)

    # Morphological dilation (A ⊕ B)
    assert cm.dilated_mask is not None
    assert cm.dilated_mask.shape == (600, 800)

    # Final combined mask and segmented regions overlay
    assert cm.mask is not None
    assert cm.region_visualization is not None
    assert len(cm.regions) >= 1
    assert cm.regions[0].area_px > 0


def test_dip_stage_6_error_level_analysis(synthetic_forged_image):
    """Test DIP Stage 6: JPEG recompression, absolute difference, amplification, and Jet colormap."""
    ela = run_ela_analysis(synthetic_forged_image, quality=95, amplification=20)
    
    assert ela.compressed_image is not None
    assert ela.compressed_image.shape == (600, 800, 3)
    
    assert ela.raw_diff is not None
    assert ela.raw_diff.shape == (600, 800, 3)
    assert ela.raw_diff.dtype == np.uint8

    assert ela.ela_image is not None
    assert ela.ela_image.shape == (600, 800, 3)

    assert ela.colormap_image is not None
    assert ela.colormap_image.shape == (600, 800, 3)
    
    assert ela.max_diff >= 0.0
    assert ela.mean_diff >= 0.0
