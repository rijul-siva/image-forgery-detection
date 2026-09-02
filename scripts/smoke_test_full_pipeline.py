"""
Smoke test for Member B's full PixelGuard forensic pipeline integration.
"""

import os
import sys
import tempfile
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.streamlit_app import run_full_pipeline, report_to_dict
from src.utils.data_types import ForensicReport

def create_synthetic_forgery_image() -> str:
    """Create a synthetic image with a copy-pasted region for testing."""
    h, w = 600, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background texture
    for y in range(h):
        for x in range(w):
            img[y, x] = [
                (x * 13 + y * 7) % 256,
                (x * 5 + y * 17) % 256,
                (x * 23 + y * 3) % 256
            ]

    # Draw a distinctive source shape (a star / circle with pattern)
    cv2.circle(img, (200, 200), 50, (0, 255, 255), -1)
    cv2.rectangle(img, (170, 170), (230, 230), (255, 0, 128), 3)

    # Copy the region around (200, 200) to destination (500, 350)
    src_region = img[130:270, 130:270].copy()
    img[280:420, 430:570] = src_region

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp.name, img)
    tmp.close()
    return tmp.name

def test_full_pipeline():
    print("--- Starting PixelGuard Full Pipeline Smoke Test ---")
    
    img_path = create_synthetic_forgery_image()
    try:
        print(f"Testing pipeline on synthetic image: {img_path}")
        report: ForensicReport = run_full_pipeline(
            img_path,
            detector="orb",
            ratio=0.75,
            ransac_threshold=3.0,
            ela_quality=95
        )
        
        # Verify PreprocessResult
        assert report.preprocess is not None
        assert report.preprocess.color_image is not None
        assert report.preprocess.gray_image is not None
        print(f"[OK] Preprocessing: Original {report.preprocess.original_width}x{report.preprocess.original_height}, Scale {report.preprocess.scale_factor:.2f}")

        # Verify CopyMoveResult
        assert report.copy_move is not None
        print(f"[OK] Copy-Move Detection: Forgery Detected = {report.copy_move.forgery_detected}, Matches = {report.copy_move.num_inlier_matches}/{report.copy_move.num_total_matches}")

        if report.copy_move.transform_params:
            tp = report.copy_move.transform_params
            print(f"     Transform: Rotation={tp.rotation_degrees:.1f} deg, Scale={tp.scale_factor:.2f}x, TransDist={tp.translation_distance:.1f}px")

        # Verify ELAResult
        assert report.ela is not None
        assert report.ela.ela_image is not None
        print(f"[OK] ELA Analysis: Suspicion Level = {report.ela.suspicion_level.upper()}")

        # Verify MetadataResult
        assert report.metadata is not None
        print(f"[OK] EXIF Metadata: Stripped = {report.metadata.metadata_stripped}, Editing Software = {report.metadata.has_editing_software}")

        # Verify JSON Export Conversion
        report_dict = report_to_dict(report)
        assert isinstance(report_dict, dict)
        assert "copy_move_detection" in report_dict
        print("[OK] JSON Export Serialization successful!")

        print("\n[SUCCESS] Smoke Test PASSED successfully!")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    test_full_pipeline()
