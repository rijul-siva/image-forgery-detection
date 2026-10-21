# PixelGuard Evaluation Report
**Detector:** `ORB`
**Date:** Automatically evaluated across 11 benchmark test images.

## 1. Detection Classification Performance

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | 81.8% | Overall correct classifications |
| **Precision** | 80.0% | TP / (TP + FP) |
| **Recall** | 100.0% | TP / (TP + FN) |
| **F1-Score** | 0.889 | Harmonic mean of precision and recall |

**Confusion Matrix:**
- True Positives (TP): `8`
- False Positives (FP): `2`
- True Negatives (TN): `1`
- False Negatives (FN): `0`

## 2. Quantitative Parameter Estimation Errors

| Parameter | Mean Error | Unit / Formulation |
| :--- | :--- | :--- |
| **Rotation Angle ($	heta$)** | 0.11° | Absolute angular error |
| **Scale Factor ($s$)** | 5.21% | Relative percentage error |
| **Translation Distance ($d$)** | 47.87 px | Euclidean distance offset |
| **Localization IoU** | 0.738 | Bounding box overlap |
| **Average Runtime** | 192.0 ms | End-to-end pipeline latency |

## 3. Detailed Per-Image Test Results

| Image | Scenario | Ground Truth | Detection | Status | Inliers / Matches | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `forged_001_clean_translation.png` | forged_001_clean_translation | Forged | Forged | **TP (Hit)** | 560/1019 | 1638.7 ms |
| `forged_002_rotation_15deg.png` | forged_002_rotation_15deg | Forged | Forged | **TP (Hit)** | 270/583 | 51.7 ms |
| `forged_003_rotation_neg25deg.png` | forged_003_rotation_neg25deg | Forged | Forged | **TP (Hit)** | 265/543 | 51.4 ms |
| `forged_004_scaled_up.png` | forged_004_scaled_up | Forged | Forged | **TP (Hit)** | 207/407 | 46.6 ms |
| `forged_005_scaled_down.png` | forged_005_scaled_down | Forged | Forged | **TP (Hit)** | 299/606 | 50.7 ms |
| `forged_006_combined_rot_scale.png` | forged_006_combined_rot_scale | Forged | Forged | **TP (Hit)** | 136/264 | 46.6 ms |
| `forged_007_blurred_copy.png` | forged_007_blurred_copy | Forged | Forged | **TP (Hit)** | 374/777 | 46.1 ms |
| `forged_008_jpeg_compressed.jpg` | forged_008_jpeg_compressed | Forged | Forged | **TP (Hit)** | 220/456 | 40.6 ms |
| `negative_authentic_001.png` | authentic_genuine | Authentic | Clean | **TN (Clean)** | 0/2 | 25.5 ms |
| `negative_repetitive_bricks_002.png` | repetitive_texture_bricks | Authentic | Forged | **FP (False Alarm)** | 111/1249 | 59.8 ms |
| `negative_repetitive_windows_001.png` | repetitive_texture_windows | Authentic | Forged | **FP (False Alarm)** | 151/1220 | 53.9 ms |
