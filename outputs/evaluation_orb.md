# PixelGuard Evaluation Report
**Detector:** `ORB`  
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
| **Rotation Angle ($	heta$)** | 0.34° | Absolute angular error |
| **Scale Factor ($s$)** | 5.07% | Relative percentage error |
| **Translation Distance ($d$)** | 1.57 px | Euclidean distance offset |
| **Localization IoU** | 0.768 | Bounding box overlap |
| **Average Runtime** | 357.7 ms | End-to-end pipeline latency |

## 3. Detailed Per-Image Test Results

| Image | Scenario | Ground Truth | Detection | Status | Inliers / Matches | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `forged_001_clean_translation.png` | forged_001_clean_translation | Forged | Forged | **TP (Hit)** | 589/599 | 506.3 ms |
| `forged_002_rotation_15deg.png` | forged_002_rotation_15deg | Forged | Forged | **TP (Hit)** | 304/333 | 226.2 ms |
| `forged_003_rotation_neg25deg.png` | forged_003_rotation_neg25deg | Forged | Forged | **TP (Hit)** | 290/299 | 259.2 ms |
| `forged_004_scaled_up.png` | forged_004_scaled_up | Forged | Forged | **TP (Hit)** | 228/248 | 112.9 ms |
| `forged_005_scaled_down.png` | forged_005_scaled_down | Forged | Forged | **TP (Hit)** | 284/347 | 152.6 ms |
| `forged_006_combined_rot_scale.png` | forged_006_combined_rot_scale | Forged | Forged | **TP (Hit)** | 168/187 | 123.8 ms |
| `forged_007_blurred_copy.png` | forged_007_blurred_copy | Forged | Forged | **TP (Hit)** | 401/423 | 184.4 ms |
| `forged_008_jpeg_compressed.jpg` | forged_008_jpeg_compressed | Forged | Forged | **TP (Hit)** | 256/287 | 147.6 ms |
| `negative_authentic_001.png` | authentic_genuine | Authentic | Clean | **TN (Clean)** | 0/1 | 42.5 ms |
| `negative_repetitive_bricks_002.png` | repetitive_texture_bricks | Authentic | Clean | **TN (Clean)** | 75/883 | 1132.5 ms |
| `negative_repetitive_windows_001.png` | repetitive_texture_windows | Authentic | Clean | **TN (Clean)** | 159/876 | 1046.7 ms |
