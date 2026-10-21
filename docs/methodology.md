# PixelGuard: Technical Methodology & Mathematical Formulation

**Course:** Image Processing
**Project:** PixelGuard — Digital Image Forgery Detection, Localization, and Parameter Estimation
**Author:** Member C (Dataset, Testing, Evaluation & Documentation Lead)

---

## 1. Architectural Overview

PixelGuard implements a multi-stage forensic analysis pipeline designed to detect, localize, and quantitatively parameterize planar copy-move image manipulations (cloning within the same image). The architecture couples classical feature matching and robust geometric estimation with supplementary forensic indicators (Error Level Analysis and EXIF metadata extraction).

```
                      +────────────────────────────────────────+
                      │           Input Image (.jpg / .png)    │
                      +───────────────────┬────────────────────+
                                          │
                                          v
                      +────────────────────────────────────────+
                      │       Stage 1: Preprocessing           │
                      │  - Validate channels & aspect ratio    │
                      │  - Resize to max_edge <= 1600 px       │
                      │  - Track scale_factor (native -> res)  │
                      │  - Grayscale conversion & denoising    │
                      +───────┬────────────────────────┬───────+
                              │                        │
               Grayscale Image│           Original File│
                              v                        v
+─────────────────────────────────────────+  +─────────────────────────────────────────+
│   Stage 2: Copy-Move Engine (Member A)  │  │   Stage 3: Supporting Forensics (B)     │
│                                         │  │                                         │
│ 1. Feature Detection (ORB / SIFT)       │  │ 1. Error Level Analysis (ELA)           │
│ 2. Same-Image Descriptor Matching (k=3) │  │    - Controlled JPEG recompression (q=95)│
│ 3. Lowe's Ratio + Spatial Suppression   │  │    - Difference amplification (alpha=20) │
│ 4. RANSAC Geometric Verification (2x3)  │  │    - Statistical suspicion classification│
│ 5. Affine Parameter Decomposition       │  │ 2. EXIF Metadata Inspection             │
│ 6. Adaptive Multi-Cluster Localization  │  │    - Camera Make / Model verification   │
│ 7. Morphological Mask Refinement        │  │    - Editing software tags (Photoshop)  │
│ 8. Region Metrics (BBox, Area, Moments) │  │    - Privacy stripping detection        │
+────────────────────┬────────────────────+  +────────────────────┬────────────────────+
                     │                                            │
                     └─────────────────────┬──────────────────────┘
                                           │
                                           v
                      +────────────────────────────────────────+
                      │       Stage 4: Consolidated Report     │
                      │  - Forgery status & confidence score   │
                      │  - Decomposed geometric parameters    │
                      │  - Visual overlay & dual coordinates   │
                      │  - Streamlit UI Dashboard presentation │
                      +────────────────────────────────────────+
```

---

## 2. Stage-by-Stage Mathematical Formulation

### 2.1 Stage 1: Preprocessing & Coordinate Tracking

High-resolution smartphone captures often exceed 12 MP ($4032 \times 3024$). Downscaling accelerates feature extraction while bounding memory consumption:

$$\text{scale\_factor} = \begin{cases}
\frac{\text{max\_edge}}{\max(W_{\text{orig}}, H_{\text{orig}})} & \text{if } \max(W_{\text{orig}}, H_{\text{orig}}) > \text{max\_edge} \\
1.0 & \text{otherwise}
\end{cases}$$

The canvas dimensions become:
$$W_{\text{res}} = \text{round}(W_{\text{orig}} \cdot \text{scale\_factor}), \quad H_{\text{res}} = \text{round}(H_{\text{orig}} \cdot \text{scale\_factor})$$

Color conversion follows standard ITU-R BT.601 luminance weighting:
$$I_{\text{gray}}(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$

---

### 2.2 Stage 2: Feature Extraction

Two complementary keypoint and descriptor extractors are supported:

1. **ORB (Oriented FAST and Rotated BRIEF):**
   - **Detector:** FAST-9 corner detector evaluated across an 8-level Gaussian image pyramid. Keypoint orientation is assigned via intensity centroid moments:
     $$m_{pq} = \sum_{x, y} x^p y^q I(x, y), \quad \theta = \text{atan2}(m_{01}, m_{10})$$
   - **Descriptor:** Steered 256-bit binary test vector, matched using bitwise XOR Hamming distance:
     $$D_{\text{Hamming}}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{256} (\mathbf{u}_i \oplus \mathbf{v}_i)$$

2. **SIFT (Scale-Invariant Feature Transform):**
   - **Detector:** Scale-space extrema in Difference-of-Gaussians ($D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)$).
   - **Descriptor:** 128-dimensional local gradient orientation histogram normalized to unit Euclidean length, matched using $L_2$ norm:
     $$D_{L2}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2$$

---

### 2.3 Stage 3: Same-Image Matching & False Match Suppression

Unlike two-image stereo matching, copy-move matching compares all descriptors within a single image against each other ($\mathbf{D} \times \mathbf{D}$).

1. **$k=3$ Nearest-Neighbour Query:**
   Because each descriptor queries the same set, the nearest match ($nn[0]$) is identically the descriptor itself at distance $0$. Hence, $k=3$ neighbours are retrieved:
   - $nn[0]$: Self-match (discarded).
   - $nn[1]$: Best external candidate match ($d_1$).
   - $nn[2]$: Second-best external candidate ($d_2$).

2. **Lowe's Ratio Test:**
   A match is accepted only if the best candidate is significantly closer than the second-best, rejecting ambiguous textures:
   $$\frac{d_1}{d_2} < \tau_{\text{ratio}} \quad (\text{default: } 0.75)$$

3. **Spatial Distance Suppression:**
   Features located in near-identical pixel coordinates describe trivial local self-similarity (e.g., adjacent pixels on a corner) and do not represent cloned objects. Pairs with spatial separation below $\tau_{\text{spatial}}$ are filtered out:
   $$\|\mathbf{p}_{\text{src}} - \mathbf{p}_{\text{dst}}\|_2 \ge \tau_{\text{spatial}} \quad (\text{default: } 10\text{ px})$$

---

### 2.4 Stage 4: Geometric Verification via RANSAC

A copy-move manipulation within a single planar photograph preserves geometric similarity (uniform scale, in-plane rotation, and 2D translation). Random false matches do not share a coherent transformation.

The transformation is modeled as a 2×3 affine similarity matrix:
$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Using `cv2.estimateAffinePartial2D` with RANSAC:
- **Minimum sample:** 2 point correspondences per iteration ($4$ DOF).
- **Inlier criterion:** Reprojection error below threshold $\tau_{\text{reproj}}$ (default: $3.0\text{ px}$):
  $$\|\mathbf{p}'_i - \mathbf{M} \tilde{\mathbf{p}}_i\|_2 \le \tau_{\text{reproj}}$$
- **Confidence threshold:** RANSAC runs up to $2000$ iterations with $99.5\%$ consensus probability.

---

### 2.5 Stage 5: Transformation Parameter Decomposition

The estimated similarity matrix $\mathbf{M}$ is decomposed into physical quantities:

$$\mathbf{M} = \begin{bmatrix} s \cos\theta & -s \sin\theta & t_x \\ s \sin\theta & s \cos\theta & t_y \end{bmatrix}$$

1. **Scale Factor ($s$):**
   $$s = \sqrt{a^2 + c^2}$$
2. **Rotation Angle ($\theta$):**
   $$\theta = \text{atan2}(c, a) \times \frac{180^\circ}{\pi} \quad (\theta \in (-180^\circ, 180^\circ])$$
3. **Algebraic Origin Translation:**
   $$\mathbf{t}_{\text{matrix}} = [t_x, t_y]^T, \quad d_{\text{matrix}} = \sqrt{t_x^2 + t_y^2}$$
4. **Physical Centroid Displacement:**
   When rotation is non-zero ($\theta \ne 0$), rotation around the object centroid $(c_x, c_y)$ creates an orbital shift of the $(0, 0)$ coordinate origin:
   $$t_x = \Delta x_{\text{phys}} + c_x - (s\cos\theta \cdot c_x - s\sin\theta \cdot c_y)$$
   Therefore, the true physical paste displacement matches the centroid-to-centroid vector:
   $$\mathbf{t}_{\text{phys}} = (\bar{x}_{\text{dst}} - \bar{x}_{\text{src}}, \bar{y}_{\text{dst}} - \bar{y}_{\text{src}}), \quad d_{\text{phys}} = \|\mathbf{t}_{\text{phys}}\|_2$$

---

### 2.6 Stage 6: Localization & Morphological Refinement

1. **Inlier Splatting:** A binary canvas is created by splatting filled circles ($\text{radius} = 24\text{ px}$) centered on verified inlier keypoints.
2. **Morphological Closing & Dilation:**
   - Morphological Closing (dilation followed by erosion with a $21 \times 21$ elliptical structuring element) bridges gaps between sparse keypoints within the cloned object.
   - Dilation ($11 \times 11$) expands boundaries to envelop the object contour.
3. **Connected Components & Region Metrics:**
   - Contours are extracted using `cv2.findContours`.
   - Area: $A = \text{cv2.contourArea}(\text{contour})$.
   - Centroids via spatial moments:
     $$\bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$$
   - Bounding boxes: $(x, y, w, h) = \text{cv2.boundingRect}(\text{contour})$.

---

### 2.7 Stage 7: Supporting Forensics (ELA & EXIF)

1. **Error Level Analysis (ELA):**
   - The image is saved in memory as a JPEG with quality $Q=95$.
   - Difference image: $\Delta(x, y) = |I_{\text{original}}(x, y) - I_{\text{recompressed}}(x, y)|$.
   - Amplification: $I_{\text{ELA}}(x, y) = \text{clip}(20 \cdot \Delta(x, y), 0, 255)$.
   - Suspicion is evaluated using statistical error distribution (mean error $\mu$ and max error).

2. **EXIF Metadata Inspection:**
   - Tag extraction via Pillow: Camera Make, Model, DateTimeOriginal, Software.
   - Keyword matching against known photo manipulation software (`photoshop`, `gimp`, `canva`, etc.).
   - Evaluation of missing EXIF headers as an informational indicator (standard on social media).
