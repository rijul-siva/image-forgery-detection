# PixelGuard — Final Project Plan

## Digital Image Forgery Detection, Localization, and Parameter Estimation

**Course:** Image Processing
**Team Size:** 3 Members
**Product Name:** PixelGuard

---

# 1. Project Overview

### Problem Statement

Digital images can be manipulated through copy-paste operations within the same image (copy-move forgery). PixelGuard analyzes uploaded images for forensic indicators, highlights suspicious regions, and **quantifies the detected manipulation** by decomposing the geometric transformation into measurable parameters (rotation angle, scale factor, translation distance, region area).

> **Scope boundary:** The system reports forensic indicators, suspicious regions, and estimated parameters. It does not claim absolute proof that an image is authentic or forged. Confidence scores are experimental, not calibrated probabilities.

### Aim

Build an image forensics system that:

1. Detects possible copy-move manipulation in an image.
2. Highlights the suspicious (copied and source) regions.
3. **Calculates quantitative parameters describing the manipulation** — rotation angle, scale factor, translation distance, region area/dimensions, and a confidence score.
4. Provides supplementary forensic signals via Error Level Analysis (ELA) and metadata inspection.
5. Presents all results through a clean, usable interface.

### Why Parameter Estimation Matters

The faculty requirement is that the project should **calculate parameters** (dimensions, angles, depth, etc. were given as examples). Copy-move forgery detection naturally supports this: once a forged region and its source are matched and geometrically verified, the transformation between them can be decomposed into real numeric quantities. This plan makes that decomposition an explicit, first-class module of the pipeline.

### Capture Constraint (Faculty Requirement)

- The **input photo must be taken with a smartphone camera** — no other capture device.
- The **forged version is created digitally** (copy-paste a region within that same photo) — done on a laptop. This simulates the manipulation the system detects.
- Analysis, processing, and UI all run on a laptop.

---

# 2. Why This Project Fits an Image Processing Course

This project involves real image-processing and computer-vision techniques instead of only using a pre-trained AI model, and it involves genuine **quantitative estimation**, not just binary classification.

### Core Concepts Covered

| Pipeline Stage | Concepts |
| :--- | :--- |
| Preprocessing | Grayscale conversion, resizing/interpolation, noise reduction |
| Feature Extraction | Keypoint detection, descriptors (ORB, SIFT), scale-space |
| Feature Matching | Brute-force/FLANN matching, Lowe's ratio test, Hamming/L2 distance |
| Geometric Verification | Affine transformations, RANSAC, inlier/outlier classification |
| Parameter Extraction | Matrix decomposition (rotation, scale, translation via `atan2`, norms) |
| Region Analysis | Binary masks, morphological operations, contour detection, connected components |
| Region Metrics | `cv2.contourArea()`, `cv2.boundingRect()`, `cv2.moments()`, centroid distance |
| Supporting Forensics | JPEG compression analysis (ELA), EXIF metadata inspection |
| Visualization | Annotated images, match overlays, highlighted regions, parameter reports |

### Practical Applications

- Digital evidence analysis
- Journalism and misinformation verification
- Insurance fraud investigation
- Academic image integrity
- General digital image forensics

---

# 3. Technology Stack

| Component | Technology |
| :--- | :--- |
| Language | Python |
| Image Processing | OpenCV |
| Feature Detection | SIFT / ORB |
| Numerical Processing | NumPy |
| Image Handling | Pillow |
| Metadata | ExifRead / Pillow |
| UI | Streamlit |
| Visualization | OpenCV / Matplotlib |
| Parameter Tabulation | Pandas |
| Testing | Pytest |
| Version Control | Git + GitHub |

### Optional Libraries

- scikit-image for additional image-processing utilities

---

# 4. High-Level System Architecture

``` text
                        SMARTPHONE
                            |
                            v
                     Capture Original Photo
                            |
                            v
                 Create Forged Version (laptop,
                 digital copy-paste within same photo)
                            |
                            v
+---------------------------------------------------------+
|                  ANALYSIS PIPELINE (laptop)              |
|                                                          |
|   Input Image                                            |
|        |                                                 |
|        v                                  [MEMBER B]     |
|   Image Preprocessing ─────────────────────────────┐     |
|   (resize, grayscale, denoise)                     |     |
|        |                                           |     |
|        v                                           |     |
|   ┌─────────────────────┐    ┌──────────────────┐  |     |
|   │     [MEMBER A]      │    │   [MEMBER B]     │  |     |
|   │                     │    │                  │  |     |
|   │  Feature Extraction │    │  Error Level     │  |     |
|   │  (SIFT / ORB)       │    │  Analysis (ELA)  │  |     |
|   │        |            │    │       |          │  |     |
|   │        v            │    │       v          │  |     |
|   │  Feature Matching   │    │  ELA Heatmap     │  |     |
|   │  (within same img)  │    │                  │  |     |
|   │        |            │    ├──────────────────┤  |     |
|   │        v            │    │                  │  |     |
|   │  False Match        │    │  Metadata / EXIF │  |     |
|   │  Filtering          │    │  Analysis        │  |     |
|   │        |            │    │       |          │  |     |
|   │        v            │    │       v          │  |     |
|   │  RANSAC Geometric   │    │  Metadata Report │  |     |
|   │  Verification       │    │                  │  |     |
|   │        |            │    └────────┬─────────┘  |     |
|   │   ┌────┴────┐       │             |            |     |
|   │   |         |       │             |            |     |
|   │   v         v       │             |            |     |
|   │ Region   Transform  │             |            |     |
|   │ Local-   Matrix     │             |            |     |
|   │ ization  Decomp.    │             |            |     |
|   │   |         |       │             |            |     |
|   │   v         v       │             |            |     |
|   │ Mask     Rotation / │             |            |     |
|   │ Refine   Scale /    │             |            |     |
|   │   |      Translation│             |            |     |
|   │   v         |       │             |            |     |
|   │ Region     |        │             |            |     |
|   │ Metrics    |        │             |            |     |
|   │   |        |        │             |            |     |
|   │   +────┬───+        │             |            |     |
|   └────────|────────────┘             |            |     |
|            |                          |            |     |
|            +─────────┬────────────────+            |     |
|                      |                             |     |
|                      v                             |     |
|          Consolidated Forensic Report              |     |
|          (detection + all parameters               |     |
|           + ELA + metadata)                        |     |
|                      |                             |     |
|                      v                             |     |
|          UI Display  ──────────────────────────────┘     |
+---------------------------------------------------------+
                            |
                            v
                  [MEMBER C: Dataset, Testing, Evaluation]
```

---

# 5. Step-by-Step Implementation Plan

## Step 1: Set Up the Project

### Repository Structure

``` text
pixelguard/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── app/
│   └── streamlit_app.py              [Member B]
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing.py              [Member B]
│   │
│   ├── copy_move/                    [Member A]
│   │   ├── __init__.py
│   │   ├── feature_extraction.py
│   │   ├── feature_matching.py
│   │   ├── geometric_verification.py
│   │   ├── parameters.py
│   │   ├── localization.py
│   │   └── pipeline.py
│   │
│   ├── forensics/                    [Member B]
│   │   ├── __init__.py
│   │   ├── ela.py
│   │   └── metadata.py
│   │
│   └── utils/                        [Shared]
│       ├── __init__.py
│       ├── image_io.py
│       ├── validation.py
│       ├── visualization.py
│       └── data_types.py
│
├── data/                             [Member C]
│   ├── original/
│   ├── forged/
│   ├── ground_truth/
│   └── test/
│
├── tests/                            [Member C]
│   ├── test_preprocessing.py
│   ├── test_copy_move.py
│   ├── test_forensics.py
│   └── test_integration.py
│
├── scripts/                          [Member C]
│   ├── evaluate.py
│   └── benchmark.py
│
├── docs/                             [Member C]
│   ├── methodology.md
│   ├── experiments.md
│   └── limitations.md
│
├── outputs/
└── results/
```

### Dependencies (`requirements.txt`)

``` text
opencv-python
opencv-contrib-python
numpy
matplotlib
streamlit
pandas
Pillow
exifread
pytest
scikit-image
```

---

## Step 2: Collect or Create a Dataset

> **Owner: Member C**, with input from all members on what test scenarios are needed.

You need two categories of images, both originating from smartphone photos:

1. **Original images** — captured directly with a smartphone.
2. **Copy-move forged images** — created by digitally copy-pasting a region within an original smartphone photo (done in software, on laptop).

### Recommended Process

1. Take a normal photo with your smartphone.
2. Select an object or region.
3. Copy the region.
4. Paste it somewhere else in the same image, applying a **known transformation** (e.g., rotate 15°, scale 1.1×, move 100 px right) so you have ground truth to compare estimated parameters against later.
5. Save the result as a forged image, and **record the transformation you applied** in a ground-truth log file.

### Ground-Truth Log Format

Create a JSON file for each forged image:

``` text
data/ground_truth/forged_001.json
{
    "source_image": "original_001.jpg",
    "forged_image": "forged_001.jpg",
    "rotation_degrees": 15.0,
    "scale_factor": 1.1,
    "translation_x_px": 100,
    "translation_y_px": -30,
    "region_description": "tree on the left"
}
```

### Variations to Create

- Simple copy-paste (no transform)
- Rotated copied regions
- Scaled copied regions
- Slightly blurred copied regions
- JPEG-compressed images
- Naturally repetitive scenes (negative test cases: windows, bricks, leaves)

This lets you test robustness and compare estimated vs. actual rotation/scale/translation.

---

## Step 3: Image Preprocessing

> **Owner: Member B** — Preprocessing is part of the input pipeline, handled by the same person who manages image upload, validation, and the UI.

Purpose: prepare the image for downstream analysis by all modules.

### Operations

1. Load and validate image
2. Resize large images (smartphone photos are often very high resolution — resize for speed while preserving aspect ratio)
3. Convert to grayscale (for feature detection)
4. Apply mild denoising if needed (Gaussian blur or non-local means)

### Pipeline

``` text
Original Image (from UI upload)
      |
      v
Validate (format, size, channels)
      |
      v
Resize (cap longest edge, e.g. 1600 px)
      |
      v
Grayscale Conversion
      |
      v
Optional Denoising
      |
      v
PreprocessResult (color_image, gray_image, scale_factor, original_dimensions)
```

Do not over-process. Excessive filtering removes useful features.

**Output:** A `PreprocessResult` dataclass containing both the color image and processed grayscale image, passed to all downstream modules.

---

## Step 4: Feature Extraction

> **Owner: Member A**

Use ORB or SIFT to find distinctive points in the image.

``` text
Grayscale Image
  |
  v
Feature Detector (ORB or SIFT)
  |
  +----> Keypoint locations
  |
  +----> Feature descriptors
```

Start with ORB (free, fast). Optionally compare with SIFT for evaluation (see Evaluation section).

---

## Step 5: Feature Matching Within the Same Image

> **Owner: Member A**

Copy-move forgery creates similar regions; features from the copied region should match features from the original region.

``` text
All descriptors
       |
       v
Feature Matcher (BFMatcher or FLANN)
       |
       v
Candidate matching pairs
```

Each candidate match records: point 1 location, point 2 location, matching distance.

---

## Step 6: Filter False Matches

> **Owner: Member A**

Images naturally contain repeated patterns. Apply filtering:

### Distance Filtering

Keep matches with sufficiently good descriptor similarity (ratio test or distance threshold).

### Spatial Distance Filtering

Remove matches where the two keypoints are too close together (self-match suppression):

``` text
If distance(point_A, point_B) < threshold:
    reject match
```

The remaining matches become candidate forgery matches.

---

## Step 7: Geometric Verification Using RANSAC

> **Owner: Member A**

A copied region typically undergoes translation, rotation, and/or scaling. Random feature matches will not follow the same transformation.

``` text
Candidate Matches
       |
       v
Estimate Transformation (Affine, via RANSAC)
       |
       +----> Outliers rejected
       |
       +----> Consistent inlier matches retained
       |
       +----> 2×3 affine transformation matrix
```

**Use `cv2.estimateAffinePartial2D()` with RANSAC** — copy-move forgeries occur within a single camera plane, so a similarity/affine transform (rotation + uniform scale + translation) is the appropriate and easily-decomposable model.

Output: inlier matches, outlier matches, and the **2×3 affine transformation matrix**.

---

## Step 8: Decompose the Transformation into Parameters

> **Owner: Member A** — This is the core "parameter calculation" step satisfying the faculty requirement.

The affine matrix `M` returned by Step 7 has the form:

``` text
M = [ a  b  tx ]
    [ c  d  ty ]
```

From this, extract:

| Parameter | Formula |
| :--- | :--- |
| **Scale factor (s)** | `s = sqrt(a² + c²)` |
| **Rotation angle (θ)** | `θ = atan2(c, a)` (convert to degrees) |
| **Translation vector (tx, ty)** | Directly from the matrix |
| **Translation distance** | `d = sqrt(tx² + ty²)` |

``` text
Affine Matrix (from RANSAC)
       |
       v
Decompose Matrix
       |
       +----> Rotation angle (degrees)
       +----> Scale factor
       +----> Translation vector (dx, dy)
       +----> Translation distance (px)
```

These numbers describe **how** the region was moved — a genuine geometric measurement.

---

## Step 9: Localize the Forged Regions

> **Owner: Member A**

1. Collect verified (inlier) keypoint locations.
2. Group nearby keypoints (clustering — simple distance-based grouping or `cv2.kmeans`).
3. Create regions around groups (convex hull or bounding box per cluster).
4. Generate a binary mask.

Both the source and copied locations should be highlighted.

---

## Step 10: Refine the Detection Mask

> **Owner: Member A**

Use morphological operations to turn scattered points into meaningful regions.

``` text
Sparse Keypoints
      |
      v
Create Mask
      |
      v
Morphological Operations (Dilation, Erosion, Opening, Closing)
      |
      v
Connected Regions
```

---

## Step 11: Compute Region Metrics

> **Owner: Member A**

From the refined mask/contours, compute additional quantitative parameters:

| Parameter | How to Compute |
| :--- | :--- |
| **Bounding box width/height (px)** | `cv2.boundingRect()` on each contour |
| **Region area (px² and % of total image)** | `cv2.contourArea()` |
| **Number of suspicious regions** | Count of connected components/contours |
| **Centroid distance (source ↔ copy)** | Euclidean distance between cluster centroids |

---

## Step 12: Error Level Analysis (ELA)

> **Owner: Member B**

ELA provides a supplementary forensic signal by analyzing JPEG compression inconsistencies.

### Pipeline

``` text
Input Image
      |
      v
Re-save as JPEG at known quality (e.g., 95%)
      |
      v
Compute pixel-wise difference (original vs. re-saved)
      |
      v
Amplify differences (contrast enhancement)
      |
      v
ELA Visualization (heatmap)
```

### Output

- ELA heatmap image
- Qualitative suspicion indicator (regions with unusually high/low error levels)

> **Limitation:** ELA is not proof of manipulation. It can be affected by resizing, screenshots, repeated compression, and social-media processing. Clearly state this in the report.

---

## Step 13: Metadata / EXIF Analysis

> **Owner: Member B**

Inspect available image metadata:

- Camera model
- Capture date/time
- GPS data (if present)
- Editing software tags
- Image format and dimensions
- Metadata availability / stripped metadata indicators

### Output

- Structured metadata dict
- Flags for suspicious indicators (e.g., editing software detected, metadata stripped)

> **Limitation:** Metadata can be removed or modified. Missing metadata is not proof of forgery.

---

## Step 14: Consolidate and Visualize Results

> **Owner: Member B** (UI), consuming outputs from **Member A** (detection) and **Member B** (ELA + metadata).

The final system should display:

### View 1: Original Image

The uploaded (smartphone-captured) image.

### View 2: Feature Matches

Lines connecting matching keypoints (inliers vs. outliers in different colors).

### View 3: Suspicious Regions

Detected areas highlighted with bounding boxes, contours, or a transparent overlay.

### View 4: ELA Visualization

Error Level Analysis heatmap.

### View 5: Metadata Report

Structured display of image metadata and any flags.

### View 6: Forensic Parameter Report

``` text
═══════════════════════════════════════════════════
                 PIXELGUARD FORENSIC REPORT
═══════════════════════════════════════════════════

Status:           Possible Copy-Move Forgery DETECTED
Confidence:       0.81 (inlier ratio — experimental score)

──── Transformation Parameters ────────────────────
Rotation:         12.4°
Scale Factor:     1.05×
Translation:      (134 px, −58 px)
Translation Dist: 146.2 px

──── Region Metrics ───────────────────────────────
Suspicious Regions: 2
Region 1 Bbox:    210 × 175 px  (6.3% of image area)
Region 2 Bbox:    205 × 168 px  (6.0% of image area)
Centroid Distance: 312 px (source ↔ copy)

──── Supporting Forensics ─────────────────────────
ELA:              Elevated error levels in Region 1
Metadata:         Edited with Adobe Photoshop (EXIF tag)

──── Verified Matches ─────────────────────────────
Total Matches:    312
Inlier Matches:   42

═══════════════════════════════════════════════════
```

Label the confidence score clearly as **experimental**.

---

## Step 15: Build the User Interface

> **Owner: Member B** — Streamlit application.

``` text
Open Application
       |
       v
Upload Image (smartphone photo)
       |
       v
Click "Analyze"
       |
       v
Run Full Pipeline (preprocessing → detection → ELA → metadata)
       |
       v
Display Results (images + parameter table + ELA + metadata)
```

### Suggested Layout

``` text
+---------------------------------------------------+
|           PixelGuard Image Forensics               |
+---------------------------------------------------+
| Upload Image   [ Choose File ]                     |
| [ Analyze Image ]                                  |
+---------------------------------------------------+
| Original  |  Feature Matches  |  Suspicious Regions|
+---------------------------------------------------+
|         ELA Visualization          |  Metadata     |
+---------------------------------------------------+
|            Forensic Parameter Report               |
| Rotation: 12.4°  | Scale: 1.05×  | Trans: 146.2px |
| Region Area: 6.3% of image                        |
| Confidence: 0.81 (experimental)                    |
+---------------------------------------------------+
```

---

# 6. Team Work Distribution

## Summary Table

| Member | Role | Contribution | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| **Member A** | Detection Algorithm Lead | **40%** | Core copy-move detection: feature extraction, matching, RANSAC, parameter decomposition, localization, region metrics |
| **Member B** | Input/Output + Forensics + Integration Lead | **40%** | Preprocessing, ELA, metadata analysis, Streamlit UI, pipeline integration/orchestration |
| **Member C** | Dataset, Testing, Evaluation + Documentation Lead | **20%** | Dataset creation with ground truth, testing, evaluation metrics, benchmarks, all documentation |

---

## Member A — 40%: Copy-Move Detection Algorithm Lead

### Responsibilities

- SIFT feature extraction implementation
- ORB feature extraction implementation
- Feature descriptor computation
- BFMatcher / FLANN matching (within the same image)
- Ratio-test and distance-based match filtering
- Spatial / self-match filtering
- RANSAC-based geometric verification (`cv2.estimateAffinePartial2D`)
- **Affine matrix decomposition → rotation, scale, translation parameters**
- False-positive reduction logic
- Keypoint clustering for region localization
- Binary mask generation
- Morphological mask refinement
- **Region metrics computation** (area, bounding box, centroid distance)
- Result visualization helpers (match drawing, region overlay)

### Files Owned

``` text
src/copy_move/
    __init__.py
    feature_extraction.py
    feature_matching.py
    geometric_verification.py
    parameters.py              ← affine decomposition + region metrics
    localization.py
    pipeline.py                ← orchestrates Steps 4–11
```

### Key Output

`CopyMoveResult` (see Integration Contracts below) — a single structured object containing all detection outputs.

### Deliverables

- Working copy-move detection pipeline callable via `run_copy_move_analysis(gray_image, color_image) → CopyMoveResult`
- SIFT vs. ORB comparison data
- Unit tests for core functions
- Technical documentation of the algorithm

---

## Member B — 40%: Input/Output + Supporting Forensics + Integration Lead

### Responsibilities

#### Preprocessing (Input Pipeline)

- Image loading and format validation
- Image resizing (preserve aspect ratio, cap resolution)
- Grayscale conversion
- Optional denoising
- Packaging into `PreprocessResult`

#### Error Level Analysis

- JPEG recompression at target quality
- Pixel-wise difference computation
- Contrast amplification
- ELA heatmap generation

#### Metadata Analysis

- EXIF extraction (ExifRead / Pillow)
- Metadata structuring and formatting
- Suspicious-indicator flags (editing software, stripped metadata)

#### User Interface (Streamlit)

- Image upload widget with validation
- "Analyze" button triggering the full pipeline
- Results display: original, matches, regions, ELA, metadata, parameter report
- Processing status / progress indicators
- Error handling for unsupported inputs

#### Integration Orchestration

- **Main pipeline function** that calls preprocessing → Member A's pipeline → ELA → metadata → aggregation
- Ensuring all modules communicate via the agreed contracts (see Section 8)
- Combining all outputs into the final forensic report

### Files Owned

``` text
src/preprocessing.py
src/forensics/
    __init__.py
    ela.py
    metadata.py
src/utils/
    image_io.py
    validation.py
    visualization.py          (shared with Member A for helpers)
app/
    streamlit_app.py
```

### Key Output

- `PreprocessResult` fed to all analysis modules
- `ELAResult` and `MetadataResult` (see Integration Contracts)
- The Streamlit application itself
- The `run_full_pipeline()` orchestration function

### Deliverables

- Working preprocessing module
- Working ELA module
- Working metadata module
- Functional Streamlit UI with integrated results
- Integration pipeline connecting all modules
- Module documentation

---

## Member C — 20%: Dataset, Testing, Evaluation + Documentation Lead

### Responsibilities

#### Dataset Preparation

- Capture original smartphone images
- Create forged images with **known, logged transformations** (ground truth)
- Organize data directory structure
- Create naturally repetitive negative-test images
- Create ground-truth JSON files for each forged image

#### Testing

Test scenarios:

- Simple copy-move (no transform)
- Rotated copied regions
- Scaled copied regions
- Blurred copied regions
- Compressed forged images
- Naturally repetitive scenes (windows, bricks, leaves — should NOT trigger detection)
- Images with no metadata
- Very high-resolution images
- Images with small forged regions

#### Evaluation

**Detection metrics** (where ground truth is available):

- True positives, false positives, false negatives
- Precision, recall, F1-score
- IoU for localization masks (if ground-truth masks are created)
- Processing time per image

**Parameter accuracy metrics** (using ground-truth logs):

- Estimated rotation vs. actual rotation (error in degrees)
- Estimated scale vs. actual scale (error in %)
- Estimated translation vs. actual translation (error in px)
- Present as a results table: estimated vs. ground-truth with error

#### Documentation

- README (installation, usage, examples)
- Architecture overview
- Dataset documentation (how images were created, ground-truth format)
- Methodology document
- Experiment log and results tables
- Limitations document
- Final demo script

### Files Owned

``` text
data/
    original/
    forged/
    ground_truth/
    test/
tests/
    test_preprocessing.py
    test_copy_move.py
    test_forensics.py
    test_integration.py
scripts/
    evaluate.py
    benchmark.py
docs/
    methodology.md
    experiments.md
    limitations.md
README.md
```

### Deliverables

- Curated dataset with ground-truth transformation logs
- Test suite (pytest)
- Evaluation report with parameter-accuracy tables
- All project documentation
- Demo script for final presentation

---

# 7. Integration Contracts — How the Pieces Connect

> **This section is critical.** Since this is your first team project, agree on these interfaces BEFORE writing any module code. Each member codes to these contracts so that integration is seamless.

## 7.1 Shared Data Structures

Define these in `src/utils/data_types.py` (created collaboratively, owned by no single member):

``` python
"""
Shared data structures for PixelGuard.
All modules produce and consume these types.
Agree on this file FIRST before writing any module code.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PreprocessResult:
    """Output of preprocessing. Input to all analysis modules."""
    color_image: np.ndarray          # Resized color image (BGR)
    gray_image: np.ndarray           # Grayscale version
    scale_factor: float              # How much the image was resized
    original_width: int
    original_height: int


@dataclass
class TransformParams:
    """Geometric transformation parameters from affine decomposition."""
    rotation_degrees: float          # Rotation angle in degrees
    scale_factor: float              # Uniform scale factor
    translation_x: float            # Translation in x (pixels)
    translation_y: float            # Translation in y (pixels)
    translation_distance: float     # sqrt(tx^2 + ty^2)


@dataclass
class RegionInfo:
    """Metrics for one suspicious region."""
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    area_px: float                   # Area in pixels squared
    area_percent: float              # Area as % of total image area
    centroid_x: float
    centroid_y: float


@dataclass
class CopyMoveResult:
    """Complete output of the copy-move detection pipeline."""
    forgery_detected: bool
    num_total_matches: int
    num_inlier_matches: int
    confidence_score: float          # Inlier ratio (experimental)
    inlier_keypoints_src: list       # List of (x, y) tuples
    inlier_keypoints_dst: list       # List of (x, y) tuples
    affine_matrix: Optional[np.ndarray]  # 2x3 matrix, None if not detected
    transform_params: Optional[TransformParams]
    regions: list                    # List of RegionInfo
    mask: Optional[np.ndarray]       # Binary mask of suspicious regions
    match_visualization: Optional[np.ndarray]   # Image with matches drawn
    region_visualization: Optional[np.ndarray]  # Image with regions highlighted


@dataclass
class ELAResult:
    """Output of Error Level Analysis."""
    ela_image: np.ndarray            # ELA heatmap visualization
    suspicion_level: str             # "low", "medium", "high"
    description: str                 # Human-readable summary


@dataclass
class MetadataResult:
    """Output of metadata/EXIF analysis."""
    metadata_dict: dict              # All extracted metadata key-value pairs
    has_editing_software: bool       # True if editing software tag found
    editing_software: Optional[str]  # Name of editing software, if found
    metadata_stripped: bool          # True if metadata seems stripped
    flags: list                      # List of suspicious indicator strings


@dataclass
class ForensicReport:
    """The final aggregated forensic report."""
    preprocess: PreprocessResult
    copy_move: CopyMoveResult
    ela: ELAResult
    metadata: MetadataResult
```

## 7.2 Module Function Signatures

Each member implements their module's public function matching these exact signatures:

### Member B — Preprocessing

``` python
# src/preprocessing.py

def preprocess_image(image_path: str, max_edge: int = 1600) -> PreprocessResult:
    """
    Load, validate, resize, convert to grayscale, and optionally denoise.
    Returns a PreprocessResult for all downstream modules.
    """
```

### Member A — Copy-Move Detection

``` python
# src/copy_move/pipeline.py

def run_copy_move_analysis(
    gray_image: np.ndarray,
    color_image: np.ndarray,
    detector: str = "orb",       # "orb" or "sift"
    ransac_threshold: float = 3.0,
    min_match_count: int = 10
) -> CopyMoveResult:
    """
    Run the full copy-move detection pipeline.
    Accepts preprocessed images, returns CopyMoveResult.
    """
```

### Member B — ELA

``` python
# src/forensics/ela.py

def run_ela_analysis(
    image_path: str,
    quality: int = 95,
    amplification: int = 20
) -> ELAResult:
    """
    Perform Error Level Analysis on the image.
    Uses the ORIGINAL file (not resized), since ELA analyzes compression.
    """
```

### Member B — Metadata

``` python
# src/forensics/metadata.py

def run_metadata_analysis(image_path: str) -> MetadataResult:
    """
    Extract and analyze EXIF/image metadata.
    """
```

### Member B — Full Pipeline Orchestration

``` python
# app/streamlit_app.py (or a separate orchestrator)

def run_full_pipeline(image_path: str, detector: str = "orb") -> ForensicReport:
    """
    Master function that runs the entire PixelGuard pipeline:
    1. Preprocess (Member B)
    2. Copy-Move Analysis (Member A)
    3. ELA (Member B)
    4. Metadata (Member B)
    5. Aggregate into ForensicReport
    """
    preprocess = preprocess_image(image_path)
    copy_move = run_copy_move_analysis(
        preprocess.gray_image,
        preprocess.color_image
    )
    ela = run_ela_analysis(image_path)
    metadata = run_metadata_analysis(image_path)

    return ForensicReport(
        preprocess=preprocess,
        copy_move=copy_move,
        ela=ela,
        metadata=metadata
    )
```

## 7.3 Integration Rules

1. **Agree on `data_types.py` first.** All three members review and approve the shared data structures before writing any module code. If a field needs to be added/changed, discuss with the team and update the shared file.

2. **Never pass raw data between modules.** Always use the dataclass types. Don't return a raw `numpy` array when a `CopyMoveResult` is expected.

3. **Each module is a black box.** Member A should never need to read Member B's ELA code to make their pipeline work, and vice versa. Only the agreed function signatures and return types matter.

4. **Test with mock data first.** Before integration, each member should test their module independently:
   - Member A: Create a simple test script that loads a test image, calls `preprocess_image()`, and feeds the result to `run_copy_move_analysis()`.
   - Member B: Create a mock `CopyMoveResult` with hardcoded values to test the UI display without needing Member A's code to be finished.
   - Member C: Write integration tests using real images once modules are ready.

5. **Use Python's `dataclasses` module.** The `@dataclass` decorator is built into Python 3.7+, no extra dependencies needed.

6. **Handle `None` gracefully.** If detection fails (e.g., no matches found), `CopyMoveResult.affine_matrix` will be `None`. The UI must handle this without crashing.

---

# 8. Integration Timeline and Milestones

## Phase 0: Planning and Setup (Day 1–2)

**All Members:**

- [ ] Finalize and approve this plan
- [ ] Create GitHub repository
- [ ] Set up branch structure
- [ ] Set up Python virtual environment
- [ ] Install dependencies
- [ ] **Collaboratively create and approve `src/utils/data_types.py`**
- [ ] Each member creates their branch

## Phase 1: Independent Module Development (Day 3–10)

Work in parallel on separate branches. No integration yet.

**Member A** (`feature/copy-move-engine`):

- [ ] Implement feature extraction (ORB + SIFT)
- [ ] Implement feature matching
- [ ] Implement false match filtering
- [ ] Implement RANSAC geometric verification
- [ ] Implement affine matrix decomposition (parameters.py)
- [ ] Implement localization and mask refinement
- [ ] Implement region metrics
- [ ] Wire everything into `pipeline.py` → returns `CopyMoveResult`
- [ ] Test with sample images

**Member B** (`feature/preprocessing`, `feature/ela`, `feature/metadata`, `feature/ui`):

- [ ] Implement preprocessing → returns `PreprocessResult`
- [ ] Implement ELA → returns `ELAResult`
- [ ] Implement metadata analysis → returns `MetadataResult`
- [ ] Build basic Streamlit UI skeleton (upload, display placeholders)
- [ ] Test each module independently

**Member C** (`feature/dataset`, `feature/testing`):

- [ ] Capture original smartphone photos (10–15 images minimum)
- [ ] Create forged versions with logged ground-truth transformations
- [ ] Create naturally repetitive negative-test images
- [ ] Set up ground-truth JSON files
- [ ] Write pytest test skeletons for all modules
- [ ] Begin writing evaluation scripts

## Phase 2: First Integration (Day 11–13)

> **Integration Day.** All members sync up.

- [ ] Merge all feature branches into `develop`
- [ ] Member B wires `run_full_pipeline()` connecting all modules
- [ ] Fix any contract mismatches (argument types, missing fields)
- [ ] Member C runs the full pipeline on the test dataset
- [ ] Fix bugs discovered during integration
- [ ] All members verify their module's output looks correct in the UI

## Phase 3: Polish and Parameter Tuning (Day 14–17)

**Member A:**

- [ ] Tune RANSAC threshold, minimum match count, spatial filter distance
- [ ] Run ORB vs. SIFT comparison across test images
- [ ] Fix any false-positive / false-negative issues

**Member B:**

- [ ] Polish Streamlit UI (layout, colors, error messages)
- [ ] Add processing status indicators
- [ ] Improve ELA visualization contrast/readability
- [ ] Ensure graceful handling of edge cases (no matches, corrupt images)

**Member C:**

- [ ] Run full evaluation: detection metrics + parameter accuracy tables
- [ ] Document successful cases and failure cases
- [ ] Write experiment log

## Phase 4: Documentation and Final Testing (Day 18–20)

**Member C (lead), all members contributing:**

- [ ] Complete README
- [ ] Complete methodology documentation
- [ ] Complete limitations documentation
- [ ] Final integration test on all dataset images
- [ ] Prepare demo script/scenario
- [ ] Final code cleanup and comments

## Phase 5: Submission

- [ ] Merge `develop` into `main`
- [ ] Tag release
- [ ] Verify everything runs from a clean clone
- [ ] Submit

---

# 9. GitHub Workflow

## Branch Strategy

``` text
main
 |
 +-- develop
      |
      +-- feature/copy-move-engine         [Member A]
      +-- feature/preprocessing            [Member B]
      +-- feature/ela                       [Member B]
      +-- feature/metadata                  [Member B]
      +-- feature/ui                        [Member B]
      +-- feature/dataset                   [Member C]
      +-- feature/testing                   [Member C]
      +-- feature/evaluation               [Member C]
```

## Rules

1. **No direct commits to `main` or `develop`.** Always use feature branches + Pull Requests.
2. Each major feature gets a dedicated branch.
3. **Pull Requests must be reviewed by at least one other member** before merging.
4. Keep commits meaningful and descriptive.

### Commit Message Examples

``` text
feat: implement ORB feature extraction
feat: add RANSAC match filtering
feat: add affine matrix decomposition (rotation, scale, translation)
feat: implement ELA visualization
feat: add metadata extraction module
feat: build Streamlit upload flow
test: add copy-move false-positive tests
test: add parameter accuracy evaluation
docs: add evaluation methodology
fix: handle images without EXIF metadata
fix: handle zero-match edge case in pipeline
```

---

# 10. Recommended Algorithm Summary

``` text
 1. Load and validate image                          [Member B]
 2. Resize if necessary                              [Member B]
 3. Convert to grayscale                             [Member B]
 4. Detect ORB/SIFT keypoints                        [Member A]
 5. Compute descriptors                              [Member A]
 6. Match descriptors within the same image          [Member A]
 7. Remove weak matches (ratio test / distance)      [Member A]
 8. Remove nearby/self matches (spatial filter)      [Member A]
 9. Apply RANSAC to estimate affine transform        [Member A]
10. Keep geometrically consistent inlier matches     [Member A]
11. Decompose affine matrix → rotation, scale, tx    [Member A]
12. Create suspicious-region mask from inliers       [Member A]
13. Apply morphology to refine mask                  [Member A]
14. Compute region metrics (area, bbox, centroid)    [Member A]
15. Run ELA on the original file                     [Member B]
16. Extract and analyze metadata                     [Member B]
17. Aggregate into forensic report                   [Member B]
18. Display in Streamlit UI                          [Member B]
19. Evaluate against ground truth                    [Member C]
```

---

# 11. Development Phases — Effort Estimates

| Phase | Description | Estimated Hours | Members |
| :--- | :--- | :--- | :--- |
| 0 | Planning, setup, contracts | 3–5 | All |
| 1 | Independent module development | 15–20 | All (parallel) |
| 2 | First integration | 4–6 | All |
| 3 | Polish and parameter tuning | 8–12 | All |
| 4 | Documentation and final testing | 6–10 | All (C leads) |
| 5 | Submission | 1–2 | All |
| **Total** | | **37–55** | |

---

# 12. Evaluation Plan

## Detection Evaluation (Member C)

### Positive Cases (should detect forgery)

- Basic copy-move
- Rotated copied regions
- Scaled copied regions
- Compressed forged images

### Negative Cases (should NOT detect forgery)

- Natural repeated objects (windows, leaves, bricks)
- Unmodified genuine images

### Detection Metrics

- True positives, false positives, false negatives
- Precision, recall, F1-score
- Processing time per image

## Parameter Accuracy Evaluation (Member C)

Since ground-truth transformations are recorded when creating forged images (Step 2), compare:

| Metric | How to Evaluate |
| :--- | :--- |
| Rotation error | |estimated θ − actual θ| (degrees) |
| Scale error | |estimated s − actual s| / actual s × 100 (%) |
| Translation error | Euclidean distance between estimated and actual translation vectors (px) |

Present as a results table:

| Image | Actual θ | Est. θ | θ Error | Actual s | Est. s | s Error | Actual d | Est. d | d Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| forged_001 | 15.0° | 14.2° | 0.8° | 1.10 | 1.08 | 1.8% | 120 px | 118 px | 2 px |
| forged_002 | 0.0° | 0.3° | 0.3° | 1.00 | 1.01 | 1.0% | 200 px | 198 px | 2 px |

This is where the project's "parameter calculation" claim becomes demonstrable and gradeable.

## ORB vs. SIFT Comparison (Member A + Member C)

| Metric | ORB | SIFT |
| :--- | :--- | :--- |
| Total keypoints detected | | |
| Matches found | | |
| Inlier matches (after RANSAC) | | |
| Rotation error (avg) | | |
| Scale error (avg) | | |
| Processing time | | |
| False positive rate | | |

---

# 13. MVP Checklist

The minimum viable product is complete when:

- [ ] User can upload an image via Streamlit
- [ ] System runs preprocessing
- [ ] System runs copy-move detection (feature extraction → matching → RANSAC)
- [ ] **System reports estimated rotation, scale, and translation parameters**
- [ ] Feature matches are visualized
- [ ] Suspicious regions are localized and highlighted
- [ ] **Region metrics are computed (area, bounding box)**
- [ ] ELA result is displayed
- [ ] Metadata can be inspected
- [ ] System produces a clear forensic parameter report
- [ ] At least one genuine-vs-forged demo scenario works reliably

---

# 14. Limitations to Mention in the Project

The system may struggle with:

- Highly smooth images with few features
- Heavy image compression
- Large scaling changes
- Strong rotations (>45°)
- Extensive post-editing (blur, color adjustment on copied region)
- Generative AI image manipulation
- Naturally repeated patterns (may cause false positives)
- Very small forged regions (too few keypoints for a stable transform estimate — this also destabilizes parameter accuracy)

ELA limitations:

- Not proof of manipulation
- Affected by resizing, screenshots, repeated compression, social-media processing

Metadata limitations:

- Can be removed or modified
- Missing metadata is not proof of forgery

This project primarily focuses on detecting and quantifying **copy-move manipulation** and should not claim to detect every type of image forgery or present its numeric parameters as forensically certified without calibration.

---

# 15. Future Improvements

## A. SIFT vs. ORB Comparison

Compare accuracy, speed, number of matches, false positives, and parameter-estimation stability between detectors.

## B. Error Level Analysis Enhancements

More sophisticated ELA with multi-quality-level analysis.

## C. Image Splicing Detection

Detect regions inserted from a different image (different scope from copy-move).

## D. Noise Inconsistency Analysis

Analyze local sensor/noise characteristics to detect spliced regions.

## E. Frequency-Domain Analysis

Use FFT, DCT, or wavelet transforms for additional forensic signals.

## F. Double JPEG Compression Detection

Identify indicators of repeated JPEG compression.

## G. Heatmap Visualization

Generate a visual map showing areas with higher suspicion, weighted by match density.

## H. Automatic Forensic Report Export

Generate a PDF/HTML report with all findings.

## I. Full Homography (Advanced)

Extend to detect perspective-distorted copies by estimating a full homography and decomposing into rotation, scale, shear, and perspective components.

## J. Deep Learning Extension

``` text
Input Image
     |
     +---------------------------+
     |                           |
Classical Forensics       Deep Learning Model
     |                           |
     +-------------+-------------+
                   |
              Evidence Fusion
                   |
            Forensic Report
```

---

# 16. Extensible Analyzer Design (Future Architecture)

For future expansion, all analysis modules should follow a common interface:

``` text
ForensicAnalyzer
    |
    +-- analyze(image)
            |
            +-- score
            +-- suspicious regions
            +-- visualizations
            +-- metadata
            +-- explanation
```

Potential future analyzers:

``` text
CopyMoveAnalyzer      (Version 1 — current)
ELAAnalyzer           (Version 1 — current)
MetadataAnalyzer      (Version 1 — current)
SplicingAnalyzer      (Version 2)
NoiseAnalyzer         (Version 2)
FrequencyAnalyzer     (Version 2)
DeepLearningAnalyzer  (Version 3)
```

---

# 17. Topics to Be Familiar With

Grouped by pipeline stage, in implementation order.

### A. Foundations

- Digital image representation: pixels, channels, color spaces (RGB, grayscale, HSV)
- Image coordinate systems (row/column vs. x/y in OpenCV)
- Python + NumPy array manipulation

### B. Preprocessing

- Image resizing / interpolation methods
- Grayscale conversion
- Noise and denoising filters (Gaussian blur, median blur)

### C. Feature Detection & Description

- Keypoints and descriptors (conceptual)
- Corner detection basics (Harris corners)
- ORB: Oriented FAST + Rotated BRIEF
- SIFT: Scale-space, difference-of-Gaussians
- Descriptor distances: Hamming (ORB) vs. L2 (SIFT)

### D. Feature Matching

- Brute-force matching vs. FLANN
- Lowe's ratio test
- Self-similarity matching within one image

### E. Geometric Transformations

- Affine transformations: rotation, scale, translation as a 2×3 matrix
- Homogeneous coordinates
- Matrix decomposition: `atan2`, matrix norms for extracting θ, s, t

### F. Robust Estimation

- RANSAC: algorithm, inliers vs. outliers
- `cv2.estimateAffinePartial2D` / `cv2.findHomography`

### G. Morphology & Region Analysis

- Binary masks, morphological operations (dilation, erosion, opening, closing)
- Connected components, contour detection
- `cv2.findContours`, `cv2.boundingRect`, `cv2.contourArea`, `cv2.moments`
- Keypoint clustering

### H. ELA & Metadata

- JPEG compression fundamentals
- Error Level Analysis concept
- EXIF metadata structure

### I. Evaluation

- Precision, recall, F1-score
- IoU for mask comparison
- Error metrics for continuous parameters (absolute error, %)

### J. Practical / Tooling

- OpenCV core API
- Matplotlib for visualization
- Streamlit basics
- Git branching and pull requests

---

# 18. Suggested Project Names and Descriptions

### Product Name

**PixelGuard**

### Academic Title

**PixelGuard: Digital Image Forgery Detection, Localization, and Geometric Parameter Estimation Using Feature Matching and RANSAC**

### Alternative Academic Title

**Copy-Move Image Forgery Detection and Geometric Transformation Parameter Estimation Using Feature Matching and RANSAC**

### Resume Description

**PixelGuard | Digital Image Forgery Detection & Parameter Estimation**

Developed a modular image-forensics system using Python and OpenCV to detect and localize copy-move image manipulation, quantify the manipulation via affine transformation decomposition (rotation, scale, translation) and region-level geometric metrics (area, bounding box, centroid distance), and provide supplementary forensic signals through Error Level Analysis and metadata inspection. Built with feature extraction, descriptor matching, spatial filtering, and RANSAC-based geometric verification.

---

# 19. Final Demo Flow

1. Open PixelGuard.
2. Upload a genuine image.
3. Show copy-move analysis (no forgery detected), ELA, and metadata.
4. Upload a known copy-move forged image.
5. Show detected suspicious regions highlighted.
6. **Show the estimated geometric parameters** (rotation, scale, translation).
7. **Compare estimated parameters against the logged ground truth.**
8. Show ELA visualization.
9. Show metadata report.
10. Explain that results are **forensic indicators**, not absolute proof.
11. Briefly mention the extensible architecture for future expansion.

---

# 20. Final Deliverables

- [ ] Source code (clean, commented, modular)
- [ ] Dataset of smartphone-captured original images and digitally forged versions
- [ ] Ground-truth transformation logs for all forged images
- [ ] Streamlit interface
- [ ] Feature match visualizations
- [ ] Forgery localization results with overlaid masks/contours
- [ ] **Parameter report per test image** (rotation, scale, translation, area, confidence)
- [ ] ELA visualizations
- [ ] Metadata reports
- [ ] **Evaluation results: estimated vs. ground-truth parameter error tables**
- [ ] ORB vs. SIFT comparison
- [ ] README documentation
- [ ] Methodology + limitations documentation
- [ ] Project presentation / final report
