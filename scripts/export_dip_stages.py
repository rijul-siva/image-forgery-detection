"""
PixelGuard: Individual DIP (Digital Image Processing) Stages Exporter.
Allows faculty and evaluators to inspect the intermediate output of every
DIP operation separately.

Saves high-resolution images for each stage into outputs/dip_stages/.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import preprocess_image
from src.copy_move.pipeline import run_copy_move_analysis
from src.forensics.ela import run_ela_analysis
from src.forensics.metadata import run_metadata_analysis


def export_dip_stages(
    image_path: str,
    output_dir: str = "outputs/dip_stages",
    detector: str = "orb",
    ratio: float = 0.75,
    ransac_threshold: float = 3.0,
    ela_quality: int = 95,
    ela_amplification: int = 20,
    boundary_extraction: bool = False,
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print("      PIXELGUARD: STEP-BY-STEP DIP PIPELINE EXPORTER")
    print("=" * 65)
    print(f"Input Image: {image_path}")
    print(f"Output Directory: {out_path.resolve()}\n")

    # -------------------------------------------------------------
    # STAGE 1: Preprocessing & Spatial Filtering
    # -------------------------------------------------------------
    print(">> [DIP Stage 1] Preprocessing & Spatial Filtering...")
    prep = preprocess_image(image_path, max_edge=1600, apply_filter=True)
    
    cv2.imwrite(str(out_path / "01_resized_color.png"), prep.color_image)
    if prep.filtered_image is not None:
        cv2.imwrite(str(out_path / "02_gaussian_filtered.png"), prep.filtered_image)
    cv2.imwrite(str(out_path / "03_grayscale_luminance.png"), prep.gray_image)
    
    print(f"   - Saved: 01_resized_color.png ({prep.color_image.shape[1]}x{prep.color_image.shape[0]})")
    print("   - Saved: 02_gaussian_filtered.png (default 5x5 Gaussian filter)")
    print("   - Saved: 03_grayscale_luminance.png (Formula: Y = 0.299R + 0.587G + 0.114B)")

    # -------------------------------------------------------------
    # STAGE 2-5: Copy-Move Detection Stages
    # -------------------------------------------------------------
    print(f"\n>> [DIP Stages 2-5] Copy-Move Forensics with Detector: {detector.upper()}...")
    cm = run_copy_move_analysis(
        prep.gray_image,
        prep.color_image,
        detector=detector,
        ratio=ratio,
        ransac_threshold=ransac_threshold,
        boundary_extraction=boundary_extraction,
    )

    # Stage 2: Feature Keypoints
    if cm.keypoints_visualization is not None:
        cv2.imwrite(str(out_path / "04_keypoints_scale_space.png"), cm.keypoints_visualization)
        print("   - Saved: 04_keypoints_scale_space.png (Rich keypoints with scale circles & orientation)")

    # Stage 3: Feature Matches
    if cm.raw_matches_visualization is not None:
        cv2.imwrite(str(out_path / "05_candidate_matches_ratio_test.png"), cm.raw_matches_visualization)
        print(f"   - Saved: 05_candidate_matches_ratio_test.png ({cm.num_total_matches} candidate matches)")

    # Stage 4: Geometric RANSAC
    if cm.match_visualization is not None:
        cv2.imwrite(str(out_path / "06_ransac_inliers_outliers.png"), cm.match_visualization)
        print(f"   - Saved: 06_ransac_inliers_outliers.png (Inliers={cm.num_inlier_matches}, Outliers={cm.num_total_matches - cm.num_inlier_matches})")

    # Stage 5: Morphology & Segmentation
    if cm.raw_mask is not None:
        cv2.imwrite(str(out_path / "07_binary_keypoint_footprint_mask.png"), cm.raw_mask)
        print("   - Saved: 07_binary_keypoint_footprint_mask.png (Raw circular keypoint footprints)")
    
    if cm.closed_mask is not None:
        cv2.imwrite(str(out_path / "08_morphological_closing.png"), cm.closed_mask)
        print("   - Saved: 08_morphological_closing.png (Adaptive elliptical closing bridges verified keypoint gaps)")

    if cm.dilated_mask is not None:
        cv2.imwrite(str(out_path / "09_morphological_dilation.png"), cm.dilated_mask)
        print("   - Saved: 09_morphological_dilation.png (Separate elliptical dilation expands the closed support)")

    if cm.coarse_region_visualization is not None:
        cv2.imwrite(str(out_path / "10_feature_match_localization.png"), cm.coarse_region_visualization)
        print("   - Saved: 10_feature_match_localization.png (Coarse localization before GrabCut)")

    if cm.region_visualization is not None:
        cv2.imwrite(str(out_path / "10_final_segmented_regions.png"), cm.region_visualization)
        stage_label = "GrabCut boundary result" if cm.boundary_extraction_enabled else "Morphological localization"
        print(f"   - Saved: 10_final_segmented_regions.png ({stage_label}, bounding boxes, translation arrow)")

    # -------------------------------------------------------------
    # STAGE 6: Error Level Analysis (ELA)
    # -------------------------------------------------------------
    print("\n>> [DIP Stage 6] Error Level Analysis (Compression Forensics)...")
    ela = run_ela_analysis(
        image_path,
        quality=ela_quality,
        amplification=ela_amplification,
        region_mask=cm.mask,
    )
    
    if ela.compressed_image is not None:
        cv2.imwrite(str(out_path / "11_ela_jpeg_resaved.png"), ela.compressed_image)
        print(f"   - Saved: 11_ela_jpeg_resaved.png (JPEG Q={ela_quality} re-compression)")

    if ela.raw_diff is not None:
        cv2.imwrite(str(out_path / "12_ela_absolute_difference.png"), ela.raw_diff)
        print("   - Saved: 12_ela_absolute_difference.png (Raw unamplified absolute error |I - I_JPEG|)")

    cv2.imwrite(str(out_path / "13_ela_amplified_heatmap.png"), ela.ela_image)
    print(f"   - Saved: 13_ela_amplified_heatmap.png (Amplified error x{ela_amplification})")

    if ela.colormap_image is not None:
        cv2.imwrite(str(out_path / "14_ela_colormap_jet.png"), ela.colormap_image)
        print("   - Saved: 14_ela_colormap_jet.png (Jet thermal colormap)")

    # Summary Report JSON
    summary = {
        "source_image": str(image_path),
        "detector": detector,
        "preprocessing": {
            "original_dimensions": [prep.original_width, prep.original_height],
            "analysis_dimensions": [prep.color_image.shape[1], prep.color_image.shape[0]],
            "scale_factor": prep.scale_factor
        },
        "copy_move": {
            "forgery_detected": cm.forgery_detected,
            "candidate_matches": cm.num_total_matches,
            "inlier_matches": cm.num_inlier_matches,
            "confidence": cm.confidence_score,
            "transform_parameters": {
                "rotation_deg": cm.transform_params.rotation_degrees if cm.transform_params else None,
                "scale_factor": cm.transform_params.scale_factor if cm.transform_params else None,
                "translation_distance_px": cm.transform_params.translation_distance if cm.transform_params else None,
            } if cm.transform_params else None,
            "regions_count": len(cm.regions),
            "boundary_extraction_enabled": cm.boundary_extraction_enabled,
            "boundary_refined_regions": cm.boundary_refined_regions,
            "closing_kernel_sizes": cm.closing_kernel_sizes,
            "dilation_kernel_sizes": cm.dilation_kernel_sizes,
        },
        "ela": {
            "suspicion_level": ela.suspicion_level,
            "mean_error": round(ela.mean_diff, 2),
            "max_error": round(ela.max_diff, 2)
        }
    }

    with open(out_path / "dip_stages_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"[SUCCESS] All DIP stages successfully exported to: {out_path.resolve()}")
    print("=" * 65 + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export individual DIP process stages.")
    parser.add_argument(
        "--image", 
        type=str, 
        default="data/forged/forged_001_clean_translation.png",
        help="Path to input image"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs/dip_stages",
        help="Directory to save stage outputs"
    )
    parser.add_argument(
        "--detector", 
        type=str, 
        choices=["orb", "sift"], 
        default="sift",
        help="Feature detector (orb or sift)"
    )
    parser.add_argument(
        "--boundary-extraction",
        action="store_true",
        help="Run GrabCut boundary refinement and export the coarse and refined DIP outputs",
    )
    args = parser.parse_args()

    # Fallback to smoke_forged.png if default image not found
    img_path = Path(args.image)
    if not img_path.exists():
        fallback = Path("outputs/smoke_forged.png")
        if fallback.exists():
            img_path = fallback
        else:
            raise FileNotFoundError(f"Input image not found: {args.image}")

    export_dip_stages(
        str(img_path),
        output_dir=args.output_dir,
        detector=args.detector,
        boundary_extraction=args.boundary_extraction,
    )


if __name__ == "__main__":
    main()
