# PixelGuard Evaluation Report
**Detector:** `SIFT`  
**Date:** Automatically evaluated across 11 benchmark test images.  

## 1. Detection Classification Performance

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | 100.0% | Overall correct classifications |
| **Precision** | 100.0% | TP / (TP + FP) |
| **Recall** | 100.0% | TP / (TP + FN) |
| **F1-Score** | 1.000 | Harmonic mean of precision and recall |

**Confusion Matrix:**
- True Positives (TP): `8`
- False Positives (FP): `0`
- True Negatives (TN): `3`
- False Negatives (FN): `0`

## 2. Quantitative Parameter Estimation Errors

| Parameter | Mean Error | Unit / Formulation |
| :--- | :--- | :--- |
| **Rotation Angle ($	heta$)** | 0.4° | Absolute angular error |
| **Scale Factor ($s$)** | 0.65% | Relative percentage error |
| **Translation Distance ($d$)** | 0.42 px | Euclidean distance offset |
| **Localization IoU** | 0.616 | Bounding box overlap |
| **Average Runtime** | 93.7 ms | End-to-end pipeline latency |

## 3. Detailed Per-Image Test Results

| Image | Scenario | Ground Truth | Detection | Status | Inliers / Matches | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `forged_001_clean_translation.png` | forged_001_clean_translation | Forged | Forged | **TP (Hit)** | 122/123 | 138.9 ms |
| `forged_002_rotation_15deg.png` | forged_002_rotation_15deg | Forged | Forged | **TP (Hit)** | 69/77 | 97.5 ms |
| `forged_003_rotation_neg25deg.png` | forged_003_rotation_neg25deg | Forged | Forged | **TP (Hit)** | 61/72 | 100.6 ms |
| `forged_004_scaled_up.png` | forged_004_scaled_up | Forged | Forged | **TP (Hit)** | 60/73 | 98.6 ms |
| `forged_005_scaled_down.png` | forged_005_scaled_down | Forged | Forged | **TP (Hit)** | 65/76 | 93.8 ms |
| `forged_006_combined_rot_scale.png` | forged_006_combined_rot_scale | Forged | Forged | **TP (Hit)** | 61/73 | 95.0 ms |
| `forged_007_blurred_copy.png` | forged_007_blurred_copy | Forged | Forged | **TP (Hit)** | 53/60 | 89.4 ms |
| `forged_008_jpeg_compressed.jpg` | forged_008_jpeg_compressed | Forged | Forged | **TP (Hit)** | 79/96 | 97.5 ms |
| `negative_authentic_001.png` | authentic_genuine | Authentic | Clean | **TN (Clean)** | 0/8 | 70.5 ms |
| `negative_repetitive_bricks_002.png` | repetitive_texture_bricks | Authentic | Clean | **TN (Clean)** | 0/0 | 76.2 ms |
| `negative_repetitive_windows_001.png` | repetitive_texture_windows | Authentic | Clean | **TN (Clean)** | 0/9 | 72.3 ms |
