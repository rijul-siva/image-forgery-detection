"""
Evaluation Engine for PixelGuard.
Designed and maintained by Member C.

Evaluates the actual detection pipeline against ground-truth annotations across:
1. Binary classification detection metrics (TP, FP, TN, FN, Precision, Recall, F1).
2. Continuous geometric parameter estimation errors (Rotation, Scale, Translation).
3. Localization IoU (Intersection-over-Union) where region ground truth is available.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.streamlit_app import run_full_pipeline
from src.utils.data_types import ForensicReport, TransformParams


DATA_DIR = REPO_ROOT / "data"
GT_DIR = DATA_DIR / "ground_truth"
FORGED_DIR = DATA_DIR / "forged"
TEST_DIR = DATA_DIR / "test"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def compute_bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union between two bounding boxes: (x, y, w, h).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - inter_area)

    return (inter_area / union_area) if union_area > 0 else 0.0


def evaluate_dataset(detector: str = "orb") -> Dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    gt_files = sorted(list(GT_DIR.glob("*.json")))

    if not gt_files:
        raise FileNotFoundError(f"No ground truth JSON files found in {GT_DIR}")

    print(f"\n===================================================================")
    print(f"       PIXELGUARD BENCHMARK EVALUATION (Detector: {detector.upper()})")
    print(f"===================================================================\n")

    results = []
    tp = fp = tn = fn = 0

    theta_errors: List[float] = []
    scale_errors: List[float] = []
    trans_errors: List[float] = []
    ious: List[float] = []
    latencies: List[float] = []

    for gt_path in gt_files:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        is_forged = gt.get("is_forged", False)

        # Locate image file
        img_name = gt.get("forged_image") or gt.get("image_name")
        img_path = FORGED_DIR / img_name if (FORGED_DIR / img_name).exists() else TEST_DIR / img_name

        if not img_path.exists():
            print(f"[WARN] Image file {img_name} not found. Skipping.")
            continue

        start_t = time.perf_counter()
        report: ForensicReport = run_full_pipeline(str(img_path), detector=detector)
        latency = (time.perf_counter() - start_t) * 1000.0  # ms
        latencies.append(latency)

        detected = report.copy_move.forgery_detected
        tp_res = report.copy_move.transform_params

        # Classification counts
        if is_forged and detected:
            tp += 1
            status = "TP (Hit)"
        elif not is_forged and not detected:
            tn += 1
            status = "TN (Clean)"
        elif not is_forged and detected:
            fp += 1
            status = "FP (False Alarm)"
        else:
            fn += 1
            status = "FN (Miss)"

        record = {
            "image": img_name,
            "scenario": gt.get("scenario", "unknown"),
            "ground_truth_forged": is_forged,
            "detected": detected,
            "status": status,
            "latency_ms": round(latency, 1),
            "inliers": report.copy_move.num_inlier_matches,
            "total_matches": report.copy_move.num_total_matches,
            "confidence": round(report.copy_move.confidence_score, 3),
        }

        # Parameter error computation for True Positives
        if is_forged and detected and tp_res is not None:
            gt_theta = gt["rotation_degrees"]
            gt_scale = gt["scale_factor"]
            gt_dist = gt["translation_distance_px"]

            # Rotation error (handling wrap/sign-invariance of affine similarity)
            # Compare absolute angles or wrapped difference
            diff_deg = abs(abs(tp_res.rotation_degrees) - abs(gt_theta))
            if diff_deg > 90.0:
                diff_deg = abs(180.0 - diff_deg)
            theta_err = diff_deg

            # Scale error (%)
            scale_err = abs(tp_res.scale_factor - gt_scale) / gt_scale * 100.0

            # Translation error (pixels)
            trans_err = abs(tp_res.translation_distance - gt_dist)

            record.update({
                "gt_theta": gt_theta,
                "est_theta": round(tp_res.rotation_degrees, 2),
                "theta_err_deg": round(theta_err, 2),
                "gt_scale": gt_scale,
                "est_scale": round(tp_res.scale_factor, 3),
                "scale_err_pct": round(scale_err, 2),
                "gt_dist_px": gt_dist,
                "est_dist_px": round(tp_res.translation_distance, 1),
                "dist_err_px": round(trans_err, 1),
            })

            theta_errors.append(theta_err)
            scale_errors.append(scale_err)
            trans_errors.append(trans_err)

            # Compute localization IoU against target region
            if "target_region_bbox" in gt and report.copy_move.regions:
                target_gt = tuple(gt["target_region_bbox"])
                best_iou = 0.0
                for r in report.copy_move.regions:
                    r_box = (r.bbox_x, r.bbox_y, r.bbox_width, r.bbox_height)
                    iou = compute_bbox_iou(r_box, target_gt)
                    if iou > best_iou:
                        best_iou = iou
                record["best_iou"] = round(best_iou, 3)
                ious.append(best_iou)

        results.append(record)

    # Compute aggregate metrics
    total_samples = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

    summary = {
        "detector": detector,
        "total_test_images": total_samples,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "classification_metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "parameter_estimation_accuracy": {
            "mean_theta_error_deg": round(float(np.mean(theta_errors)), 2) if theta_errors else None,
            "mean_scale_error_pct": round(float(np.mean(scale_errors)), 2) if scale_errors else None,
            "mean_trans_error_px": round(float(np.mean(trans_errors)), 2) if trans_errors else None,
            "mean_localization_iou": round(float(np.mean(ious)), 3) if ious else None,
        },
        "performance": {
            "mean_latency_ms": round(float(np.mean(latencies)), 1) if latencies else None
        },
        "detailed_results": results
    }

    # Print summary
    print(f"Classification Metrics:")
    print(f"  Accuracy:  {accuracy * 100:.1f}%")
    print(f"  Precision: {precision * 100:.1f}%")
    print(f"  Recall:    {recall * 100:.1f}%")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"\nParameter Estimation Errors (on True Positives):")
    if theta_errors:
        print(f"  Mean Rotation Error:    {np.mean(theta_errors):.2f}°")
        print(f"  Mean Scale Error:       {np.mean(scale_errors):.2f}%")
        print(f"  Mean Translation Error: {np.mean(trans_errors):.1f} px")
    if ious:
        print(f"  Mean Localization IoU:  {np.mean(ious):.3f}")
    if latencies:
        print(f"  Mean Latency:           {np.mean(latencies):.1f} ms")

    # Save to JSON
    out_json = OUTPUTS_DIR / f"evaluation_{detector}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\n[OK] Evaluation report saved to {out_json}")

    # Generate Markdown Table Report
    md_report = generate_markdown_report(summary)
    out_md = OUTPUTS_DIR / f"evaluation_{detector}.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[OK] Evaluation Markdown report saved to {out_md}\n")

    return summary


def generate_markdown_report(summary: Dict) -> str:
    lines = [
        f"# PixelGuard Evaluation Report",
        f"**Detector:** `{summary['detector'].upper()}`  ",
        f"**Date:** Automatically evaluated across {summary['total_test_images']} benchmark test images.  ",
        "",
        "## 1. Detection Classification Performance",
        "",
        "| Metric | Score | Note |",
        "| :--- | :--- | :--- |",
        f"| **Accuracy** | {summary['classification_metrics']['accuracy'] * 100:.1f}% | Overall correct classifications |",
        f"| **Precision** | {summary['classification_metrics']['precision'] * 100:.1f}% | TP / (TP + FP) |",
        f"| **Recall** | {summary['classification_metrics']['recall'] * 100:.1f}% | TP / (TP + FN) |",
        f"| **F1-Score** | {summary['classification_metrics']['f1_score']:.3f} | Harmonic mean of precision and recall |",
        "",
        "**Confusion Matrix:**",
        f"- True Positives (TP): `{summary['confusion_matrix']['true_positives']}`",
        f"- False Positives (FP): `{summary['confusion_matrix']['false_positives']}`",
        f"- True Negatives (TN): `{summary['confusion_matrix']['true_negatives']}`",
        f"- False Negatives (FN): `{summary['confusion_matrix']['false_negatives']}`",
        "",
        "## 2. Quantitative Parameter Estimation Errors",
        "",
        "| Parameter | Mean Error | Unit / Formulation |",
        "| :--- | :--- | :--- |",
        f"| **Rotation Angle ($\theta$)** | {summary['parameter_estimation_accuracy']['mean_theta_error_deg']}° | Absolute angular error |",
        f"| **Scale Factor ($s$)** | {summary['parameter_estimation_accuracy']['mean_scale_error_pct']}% | Relative percentage error |",
        f"| **Translation Distance ($d$)** | {summary['parameter_estimation_accuracy']['mean_trans_error_px']} px | Euclidean distance offset |",
        f"| **Localization IoU** | {summary['parameter_estimation_accuracy']['mean_localization_iou']} | Bounding box overlap |",
        f"| **Average Runtime** | {summary['performance']['mean_latency_ms']} ms | End-to-end pipeline latency |",
        "",
        "## 3. Detailed Per-Image Test Results",
        "",
        "| Image | Scenario | Ground Truth | Detection | Status | Inliers / Matches | Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in summary["detailed_results"]:
        gt_label = "Forged" if r["ground_truth_forged"] else "Authentic"
        det_label = "Forged" if r["detected"] else "Clean"
        lines.append(
            f"| `{r['image']}` | {r['scenario']} | {gt_label} | {det_label} | **{r['status']}** | {r['inliers']}/{r['total_matches']} | {r['latency_ms']} ms |"
        )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    det = sys.argv[1] if len(sys.argv) > 1 else "orb"
    evaluate_dataset(detector=det)
