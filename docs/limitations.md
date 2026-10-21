# PixelGuard: Technical Limitations & Failure Modes

**Course:** Image Processing
**Project:** PixelGuard — Digital Image Forgery Detection, Localization, and Parameter Estimation
**Author:** Member C (Dataset, Testing, Evaluation & Documentation Lead)

---

## 1. Scope Boundary & Legal Disclaimer

PixelGuard is an experimental **forensic decision-support system**. It identifies probabilistic forensic anomalies, localizes suspicious regions, and quantifies geometric transformation parameters using classical computer vision. It does **not** claim absolute legal certification that an uploaded photograph is authentic or forged.

---

## 2. Algorithmic & Geometric Limitations

### 2.1 The Smooth / Featureless Inpainting Blind Spot
- **The Issue:** The most prevalent real-world copy-move manipulation is object removal (e.g., cloning patches of green lawn over litter, clear blue sky over birds, calm water over boats, or smooth skin retouching).
- **Why Classical Keypoint Detectors Fail:** Feature detectors such as SIFT and ORB detect extrema in scale-space and high-frequency corner gradients (FAST / Harris). On smooth, low-gradient surfaces, virtually **zero keypoints** are generated.
- **Remediation:** Detection of smooth cloned regions requires dense block-based matching (e.g., Discrete Cosine Transform [DCT] or Discrete Wavelet Transform [DWT] block sorting) or deep-learning feature representations.

---

### 2.2 Planar Similarity vs. 3D Perspective Foreshortening
- **The Issue:** PixelGuard utilizes `cv2.estimateAffinePartial2D`, which models a 4-DOF similarity transformation (uniform scale $s$, 2D in-plane rotation $\theta$, and translation $[t_x, t_y]$).
- **Failure Mode:** If a cloned object is inserted into a receding depth plane (e.g., duplicating a parked car further down an angled highway, or cloning a window onto a slanted wall), 3D perspective foreshortening occurs.
- **Result:** Under perspective distortion, non-uniform scaling and out-of-plane tilting violate the planar similarity assumption. RANSAC will classify these correspondences as non-consensus outliers and fail to detect the manipulation.
- **Remediation:** Full $3 \times 3$ projective homography estimation (`cv2.findHomography`), decomposed via polar or SVD methods.

---

### 2.3 Naturally Repetitive Architectural Textures (False Positives)
- **The Issue:** Authentic photographs of building facades, apartment balconies, brick masonry, tiled walkways, and fence pickets contain identical physical features at regular spatial intervals.
- **Empirical Demonstration:** In our benchmark evaluation, the binary FAST detector in ORB identified identical feature clusters across repetitive window grids and brick patterns, yielding a false positive detection ($s = 1.0, \theta = 0^\circ$).
- **Remediation:** Repetitive lattice suppression via spatial periodicity filtering, collinear match pruning, or utilizing SIFT's more discriminative 128-D orientation histograms.

---

### 2.4 Rotational Symmetry Ambiguity
- **The Issue:** Symmetrical structures (coins, checkerboards, circular vehicle wheels, manhole covers) exhibit identical descriptor distributions across multiple rotational increments ($90^\circ, 180^\circ$).
- **Result:** RANSAC may lock onto valid mathematical correspondences that exhibit $\pm 90^\circ$ or $180^\circ$ phase shifts relative to the human-perceived orientation.

---

### 2.5 Extreme Scale and Rotation Bounds
- Scale variations exceeding $2.0\times$ or below $0.5\times$ severely degrade keypoint descriptor matching reliability.
- Out-of-plane rotations ($> 30^\circ$) cause significant photometric and geometric deformation of local patches.

---

## 3. Error Level Analysis (ELA) Limitations

### 3.1 Sharp Contrast Edge Artifacts
- In lossy JPEG compression, Discrete Cosine Transform (DCT) $8 \times 8$ blocks quantize high-frequency components aggressively.
- High-contrast natural boundaries (e.g., dark tree branches against an overexposed sky, sharp building silhouettes, black text on white paper) naturally produce elevated reconstruction errors.
- **Naive thresholding (`max > 200`) will falsely flag almost 100% of authentic smartphone photos.** ELA must be interpreted as relative error variance, not standalone proof.

### 3.2 Format & Recompression Fragility
- Lossless formats (PNG, BMP, TIFF) exhibit uniform baseline errors when re-compressed to JPEG, obscuring historical compression discrepancies.
- Multi-generation social media transcoding (e.g., uploading to WhatsApp, then downloading and editing) normalizes compression artifacts across the canvas, masking copy-move operations.

---

## 4. Metadata Forensics Limitations

### 4.1 Social Media Privacy Stripping
- Mainstream communication and social platforms (WhatsApp, Telegram, Instagram, Twitter, iMessage) automatically strip all EXIF metadata (camera serial numbers, GPS coordinates, capture dates) to safeguard user privacy.
- Therefore, **missing or stripped metadata is standard informational behavior**, not proof of malicious tampering.

### 4.2 Trivial Header Countermeasures
- EXIF tags reside in plain-text header sections and can be easily fabricated or erased using simple tools (`exiftool -all=`). Authenticity cannot rely solely on metadata integrity.
