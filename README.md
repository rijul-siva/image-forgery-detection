# PixelGuard: Digital Image Forgery Detection, Localization, and Parameter Estimation

- **Course:** Image Processing
- **Team Size:** 3 Members
- **Product Name:** PixelGuard

---

## 1. Project Overview

PixelGuard is a modular image-forensics system built with Python and OpenCV. It analyzes digital images to detect copy-move manipulation (cloning a region within the same photo), localizes both the source and replicated regions with bounding contours, and **quantifies the detected manipulation by decomposing the geometric transformation into measurable numeric parameters**:
- **Rotation Angle ($\theta$):** In-plane angular rotation in degrees.
- **Scale Factor ($s$):** Relative enlargement or reduction ratio.
- **Physical Translation Distance ($d_{\text{phys}}$):** Centroid-to-centroid Euclidean displacement in pixels.
- **Algebraic Origin Shift ($t_x, t_y$):** Coordinate shift of the $(0, 0)$ top-left origin.
- **Region Metrics:** Bounding boxes, contour area in pixels, area percentage of image, and centroids.
- **Supporting Forensics:** Error Level Analysis (ELA) heatmap and EXIF metadata inspection (camera make/model and editing software signatures).

> **Scope Boundary:** PixelGuard is a forensic decision-support system. It computes quantitative measurements and highlights anomalies; confidence scores are experimental and do not constitute absolute legal certification of image authenticity.

---

## 2. System Architecture

```
                        SMARTPHONE
                            │
                            ▼
                  Capture Original Photo
                            │
                            ▼
              Digitally Manipulated Version (Laptop)
                            │
                            ▼
+───────────────────────────────────────────────────────────────+
│                 PIXELGUARD ANALYSIS PIPELINE                  │
│                                                               │
│   Input Image (.jpg / .png)                                   │
│        │                                                      │
│        ▼                                       [Member B]     │
│   Preprocessing (Aspect resize, grayscale) ─────────────┐     │
│        │                                                │     │
│        ▼                                                │     │
│   ┌───────────────────────────┐    ┌─────────────────┐  │     │
│   │       [Member A]          │    │   [Member B]    │  │     │
│   │                           │    │                 │  │     │
│   │  Feature Detection        │    │  Error Level    │  │     │
│   │  (ORB / SIFT)             │    │  Analysis (ELA) │  │     │
│   │        │                  │    │       │         │  │     │
│   │        ▼                  │    │       ▼         │  │     │
│   │  Feature Matching (k=3)   │    │  ELA Heatmap    │  │     │
│   │        │                  │    │                 │  │     │
│   │        ▼                  │    ├─────────────────┤  │     │
│   │  False Match Filtering    │    │                 │  │     │
│   │  (Lowe's ratio + spatial) │    │  EXIF Metadata  │  │     │
│   │        │                  │    │  Analysis       │  │     │
│   │        ▼                  │    │       │         │  │     │
│   │  RANSAC Verification      │    │       ▼         │  │     │
│   │  (2x3 Affine Similarity)  │    │  Metadata Table │  │     │
│   │        │                  │    │  & Audit Flags  │  │     │
│   │   ┌────┴─────┐            │    └────────┬────────┘  │     │
│   │   ▼          ▼            │             │           │     │
│   │ Region    Affine Matrix   │             │           │     │
│   │ Localize  Decomposition   │             │           │     │
│   │   │       (θ, s, tx, ty)  │             │           │     │
│   │   ▼          │            │             │           │     │
│   │ BBox, Area,  │            │             │           │     │
│   │ Centroids    │            │             │           │     │
│   │   │          ▼            │             │           │     │
│   │   └────┬─────┘            │             │           │     │
│   └────────┼──────────────────┘             │           │     │
│            │                                │           │     │
│            └───────────────┬────────────────┘           │     │
│                            │                            │     │
│                            ▼                            │     │
│                Consolidated Forensic Report             │     │
│                            │                            │     │
│                            ▼                            │     │
│                Streamlit UI Presentation ───────────────┘     │
+───────────────────────────────────────────────────────────────+
                            │
                            ▼
               [Member C: Dataset, Testing, Evaluation]
```

---

## 3. Team Responsibilities & Work Breakdown

| Member | Role | Responsibilities |
| :--- | :--- | :--- |
| **Member A** | Detection Algorithm Lead (40%) | Core copy-move engine (`src/copy_move/`): SIFT & ORB feature extraction, matching ($k=3$), Lowe's ratio test, spatial suppression, RANSAC geometric verification, affine decomposition (`parameters.py`), localization & morphology (`localization.py`), and visualization overlays. |
| **Member B** | Forensics, UI & Integration Lead (40%) | Image preprocessing (`src/preprocessing.py`), Error Level Analysis (`src/forensics/ela.py`), EXIF metadata inspection (`src/forensics/metadata.py`), Streamlit application (`app/streamlit_app.py`), and master pipeline integration. |
| **Member C** | Dataset, Testing & Evaluation Lead (20%) | Benchmark dataset & ground truth logs (`data/`, `scripts/generate_dataset.py`), comprehensive pytest suite (`tests/`), automated evaluation engine (`scripts/evaluate.py`), ORB vs. SIFT comparative benchmarks (`scripts/benchmark.py`), and all project documentation (`docs/`, `README.md`). |

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+ or Python 3.11
- Operating System: Windows, macOS, or Linux

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/image-forgery-detection.git
   cd image-forgery-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install declared dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 5. Usage & Verification

### A. Run the Interactive Web Application (Streamlit)
Launch the PixelGuard forensics dashboard:
```bash
streamlit run app/streamlit_app.py
```
- Upload any `.jpg` or `.png` photograph.
- Select the detector (`ORB` for speed, `SIFT` for high precision).
- Click **Analyze Image** to view synchronized feature matches, localized regions, parameter reports, ELA heatmaps, and EXIF metadata.
- Export findings as structured JSON.

### B. Run the Automated Test Suite (Pytest)
Execute all 32 unit and integration tests:
```bash
pytest -v
```

### C. Run Dataset Generation
Re-generate the reproducible synthetic benchmark dataset with ground-truth logs:
```bash
python scripts/generate_dataset.py
```

### D. Run Benchmark & Evaluation Engines
Evaluate detection metrics and quantitative parameter errors across all dataset images:
```bash
# Evaluate ORB
python scripts/evaluate.py orb

# Evaluate SIFT
python scripts/evaluate.py sift

# Run Head-to-Head Comparative Benchmark
python scripts/benchmark.py
```

---

## 6. Experimental Benchmark Highlights

Evaluated across the 11-image benchmark dataset (8 positive copy-move scenarios, 3 negative architectural controls):

| Metric | ORB | SIFT | Notes |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 81.8% | **100.0%** | SIFT successfully rejects repetitive textures |
| **Recall** | **100.0%** | **100.0%** | Both detect 100% of copy-move cases |
| **Precision** | 80.0% | **100.0%** | SIFT eliminates architectural false positives |
| **Mean Rotation Error** | 0.11° | **0.03°** | Sub-degree rotational accuracy |
| **Mean Scale Error** | 5.21% | **3.12%** | High scale decomposition stability |
| **Mean Pipeline Latency** | ~192 ms | **~73 ms** | Fast, real-time performance |

---

## 7. Documentation Index

- [`docs/methodology.md`](docs/methodology.md): Full mathematical formulation of preprocessing, matching, RANSAC, affine decomposition, and morphology.
- [`docs/experiments.md`](docs/experiments.md): Comprehensive test results, parameter comparison tables, and benchmark data.
- [`docs/limitations.md`](docs/limitations.md): Detailed analysis of algorithmic blind spots (smooth inpainting, perspective distortion, repetitive patterns).
- [`PixelGuard_Final_Plan.md`](PixelGuard_Final_Plan.md): Master team specification and course plan.
