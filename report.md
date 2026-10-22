# PixelGuard — Member A Development Report

This file is a running changelog for Member A (copy-move detection pipeline).
It is intentionally **not committed or pushed** to Git — it exists locally so other
agents/team members can read what was done, why, and how it integrates.

> Audience: Member B (integration / UI), Member C (tests / dataset / evaluation),
> and any future agent taking over Member A's code.

---

## 0. Conventions

- Each commit message is recorded in section **"Commit Log"** with a short note.
- Each significant change is recorded in section **"Change Log"** with the file(s)
  touched, what it does, and how it integrates.
- "How it works end-to-end" lives in section **"Architecture"**.
- "How other modules integrate" lives in section **"Integration Guide"**.

---

## 1. Commit Log

| # | SHA  | Message | Purpose |
|---|------|---------|---------|
| 1 | `0f33492` | `chore: baseline Member B modules (preprocessing, ELA, metadata, UI)` | Brought Member B's modules onto `feature/member-a-copy-move` so the branch is integration-ready. |
| 2 | `192452d` | `chore: add requirements.txt for PixelGuard dependencies` | Pins Python deps (OpenCV contrib for SIFT, Streamlit, etc.). |
| 3 | `22c8387` | `feat(copy-move): add feature_extraction module (ORB + SIFT wrapper)` | `extract_features(gray, detector) -> FeatureSet`. |
| 4 | `55a94de` | `feat(copy-move): add feature_matching module with Lowe's ratio and spatial filter` | `match_features(features) -> List[MatchPair]`. |
| 5 | `4183e17` | `feat(copy-move): add RANSAC geometric verification and affine matrix decomposition` | `verify_geometric` + `decompose_affine` (+ `reconstruct_affine`). |
| 6 | `38a41ca` | `feat(copy-move): add localization (clustering, morphology, region metrics)` | `localize_forgery` + `centroid_distance`. |
| 7 | `06212a9` | `feat(copy-move): add visualization helpers (matches, regions, arrow)` | `draw_matches`, `draw_regions`, `draw_transformation_arrow`. |
| 8 | `d32a616` | `feat(copy-move): add pipeline entry point + wire into Streamlit UI` | `run_copy_move_analysis` + Streamlit integration. |
| 9 | (part of `d32a616`) | _same commit — see Change Log 9_ | Replaced mock with real pipeline; expanded Copy-Move tab. |
| 10 | `ce8bcf2` | `fix(copy-move): self-match suppression broke ratio test; add synthetic smoke test` | Bug fix in `feature_matching.py`; adds `scripts/smoke_test_copy_move.py`. |

---

## 2. Change Log

### Commit 1 — baseline Member B modules
- Brought in: `app/streamlit_app.py`, `src/__init__.py`, `src/preprocessing.py`, `src/forensics/ela.py`, `src/forensics/metadata.py`, `src/utils/__init__.py`, `src/utils/data_types.py`.
- These define the **integration contract** that Member A must produce and consume:
  - `preprocess_image(image_path) -> PreprocessResult` (provides `color_image`, `gray_image`).
  - `CopyMoveResult` dataclass (in `src/utils/data_types.py`) — Member A's pipeline must return this.
  - The UI currently has a `mock_run_copy_move_analysis` placeholder that Member A will replace.

### Commit 2 — requirements.txt
- Added `opencv-contrib-python` (needed for SIFT in evaluation, ORB is in base).
- Other libs already implicit in Member B's code: `streamlit`, `Pillow`, `numpy`, `opencv-python`.

### Commit 3 — `src/copy_move/__init__.py` + `feature_extraction.py`
- **Package marker** for the copy-move subpackage.
- **`extract_features(gray_image, detector='orb') -> FeatureSet`**:
  - Wraps OpenCV's ORB/SIFT detectors with a uniform interface.
  - Returns keypoints (list of `cv2.KeyPoint`) and descriptors (NxD ndarray).
  - Validates input is a 2-D uint8 image; coerces dtype if needed.
  - Raises a clear error if SIFT is requested but `opencv-contrib-python` is missing.
  - **Integration:** Called from `pipeline.py` after `preprocess_image()` produces the gray image.

### Commit 4 — `feature_matching.py`
- **`match_features(features, ratio=0.75, spatial_distance_min=10.0) -> List[MatchPair]`**:
  - Builds BFMatcher (Hamming) for ORB or FLANN KD-tree for SIFT.
  - Self-match suppression (queryIdx == trainIdx never passes).
  - Lowe's ratio test on k=2 nearest neighbours (default 0.75).
  - Spatial-distance suppression: drops pairs whose two keypoints are within
    `spatial_distance_min` pixels (these are typically trivial local
    self-similarity and add noise to RANSAC).
  - Returns a list of `MatchPair` objects carrying (src_xy, dst_xy, distance).
- **Integration:** Output feeds into `geometric_verification.py` (RANSAC).

### Commit 5 — `geometric_verification.py` + `parameters.py`
- **`verify_geometric(matches, ransac_threshold=3.0, min_match_count=10, ...) -> GeometricResult`**:
  - Calls `cv2.estimateAffinePartial2D` (similarity transform: rotation +
    uniform scale + translation) — the physically correct model for copy-move.
  - Splits matches into inliers/outliers via the RANSAC mask.
  - Returns `None` matrix if inliers are too few (< min_match_count / 2),
    so the pipeline can show "no forgery detected".
- **`decompose_affine(M) -> TransformParams`**:
  - `scale = sqrt(a^2 + c^2)`
  - `rotation = atan2(c, a)` in degrees, wrapped to (-180, 180]
  - `translation = (tx, ty)` and `translation_distance = hypot(tx, ty)`
  - **`reconstruct_affine(params) -> M`** is included for round-trip
    self-testing.
- **Integration:** Output feeds `localization.py` (uses inlier points) and
  `pipeline.py` (puts `TransformParams` and the matrix into `CopyMoveResult`).

### Commit 6 — `localization.py`
- **`localize_forgery(inlier_src, inlier_dst, image_shape) -> LocalizationResult`**:
  - Greedy 2-cluster split (farthest-point seeding + iterative refinement).
  - Per-cluster mask via filled circles at each inlier keypoint.
  - Morphological close + dilate + largest-connected-component keeps.
  - Per-region `RegionInfo` (bbox, area px, area %, centroid) computed.
- **`centroid_distance(regions)`** helper for the source↔copy offset.
- **Integration:** Output is folded into `CopyMoveResult.regions` and
  `CopyMoveResult.mask` by `pipeline.py`.

### Commit 7 — `visualization.py` (inside copy_move)
- `draw_matches(color_image, matches, inlier_mask)`: inliers green, outliers red, capped at 200 lines for readability.
- `draw_regions(color_image, regions, alpha=0.35)`: semi-transparent mask overlay + bbox + centroid + area% label.
- `draw_transformation_arrow(color_image, regions)`: white arrow from larger to smaller region centroid.
- **Why inside copy_move and not in src/utils/visualization.py?**
  Removes a cross-team runtime dependency for Member A. Member B's UI can
  import these directly or just display the pre-rendered
  `CopyMoveResult.match_visualization` / `region_visualization`.

### Commit 8 — `pipeline.py` (the public entry point)
- **`run_copy_move_analysis(gray_image, color_image, **kwargs) -> CopyMoveResult`**:
  - Single function Member B's UI calls.
  - Tunable defaults live in `pipeline.DEFAULTS` (detector, ratio,
    spatial_distance_min, ransac_threshold, min_match_count,
    forgery_min_inliers, forgery_min_inlier_ratio).
  - `forgery_detected` requires BOTH `inliers >= forgery_min_inliers`
    AND `inlier_ratio >= forgery_min_inlier_ratio` (avoids spurious flags
    on naturally repetitive scenes).
  - Returns a non-`forgery_detected` `CopyMoveResult` (never raises) on
    any internal failure, so the UI always has something to render.

### Commit 9 — wire pipeline into Member B's Streamlit app
- Replaced the `mock_run_copy_move_analysis` placeholder in
  `app/streamlit_app.py` with the real `run_copy_move_analysis`.
- Expanded the "Copy-Move" tab to actually display Member A's outputs:
  forgery flag, inlier/total counts + experimental confidence, three
  metric cards (rotation, scale, translation distance), region table
  (bbox, area %, centroid), and side-by-side match + region
  visualizations.
- Switched deprecated `use_column_width=True` to `use_container_width=True`.
- Removed unused imports (`numpy`, `CopyMoveResult`, `TransformParams`,
  `RegionInfo`, `PreprocessResult`).

### Commit 10 — fix `feature_matching.py` self-match suppression
- **Bug found by smoke test:** the original implementation called
  `knnMatch(descriptors, descriptors, k=2)`, then dropped every match
  pair where `queryIdx == trainIdx`. But with k=2, the self-match sits
  at index 0 and the *real* best candidate sits at index 1 — so the
  ratio test was being run against the self-match (distance 0) and
  always failed. Result: 0 surviving matches on any image.
- **Fix:** request `k=3`, skip the self-match, and run the ratio test
  between the real best candidate (nn[1]) and the second-best (nn[2]).
- After fix: a textured 1200x800 synthetic image yields ~124 ORB
  matches (vs 0 before); the full pipeline reports `forgery_detected=True`
  on the synthetic forgery.

### Commit 11 — `scripts/smoke_test_copy_move.py` (synthetic forgery self-test)
- Builds a controlled synthetic forgery:
  1. Smooth gray background (no features outside the objects).
  2. Place an asymmetric, photo-realistic object A at (100, 120).
  3. Warp the whole canvas by `+12deg`, scale 1.10 around A's center.
  4. Paste an identical (unwarped) copy of A at offset (+320, +140).
- Runs `run_copy_move_analysis` with both ORB and SIFT.
- Reports ground truth (from the *inverse* of the warp, since RANSAC
  estimates the unwarped-B-from-warped-A transform), the estimate,
  and the absolute errors. Sign-agnostic theta comparison.
- **Current result:**
  - ORB  : theta_err=0.01°, scale_err=0.08%, dist_err≈55 px on ~350 px GT
  - SIFT : theta_err=0.01°, scale_err≈0%,    dist_err≈55 px
- Exits 0 on pass, 1 on fail. Use this as a developer sanity check
  before pushing pipeline changes:
  `$env:PYTHONPATH=(Resolve-Path .).Path; python scripts/smoke_test_copy_move.py`

---

## 3. Architecture (how Member A's pieces fit together)

```
   PreprocessResult           (from src/preprocessing.py, Member B)
        |
        | gray_image, color_image
        v
   +---------------------------------------------------------------+
   |  src/copy_move/pipeline.py::run_copy_move_analysis            |
   |                                                               |
   |   1. extract_features(gray)  ->  FeatureSet                   |
   |       ORB or SIFT                                             |
   |                                                               |
   |   2. match_features(features)  ->  List[MatchPair]            |
   |       BFMatcher (Hamming) or FLANN KD-tree                    |
   |       Lowe's ratio (k=3), self-match suppression,             |
   |       spatial-distance suppression                            |
   |                                                               |
   |   3. verify_geometric(matches)  ->  GeometricResult           |
   |       cv2.estimateAffinePartial2D with RANSAC                 |
   |       (similarity: rotation + uniform scale + translation)    |
   |                                                               |
   |   4. decompose_affine(M)  ->  TransformParams                 |
   |       scale = sqrt(a^2 + c^2)                                 |
   |       theta = atan2(c, a)  (degrees, wrapped to (-180, 180])  |
   |       tx, ty = matrix translation column                      |
   |       translation_distance = hypot(tx, ty)                    |
   |                                                               |
   |   5. localize_forgery(inlier_src, inlier_dst)                  |
   |       greedy 2-cluster by farthest-point seeding              |
   |       per-cluster mask + morphology (close, dilate) +         |
   |       keep-largest-connected-component                         |
   |       per-region bbox / area / centroid  -> RegionInfo        |
   |                                                               |
   |   6. draw_matches + draw_regions + draw_transformation_arrow  |
   |       (numpy BGR images for the UI)                           |
   |                                                               |
   |   7. forgery_detected  iff  num_inliers >= min AND            |
   |                           inlier_ratio >= ratio_threshold     |
   |   confidence_score       =  inlier_ratio  (experimental)      |
   +---------------------------------------------------------------+
        |
        v
   CopyMoveResult    (dataclass in src/utils/data_types.py)
```

### Module dependency graph (no circular deps)

```
src.utils.data_types  <-- shared contract, owned by everyone
        ^
        |
   src.copy_move.feature_extraction
        |
   src.copy_move.feature_matching
        |
   src.copy_move.geometric_verification
        |
   src.copy_move.parameters
        |
   src.copy_move.localization
        |       \
        |        src.copy_move.visualization (independent)
        v        v
   src.copy_move.pipeline    (single public entry point)
```

### Tunable knobs (all overridable per-call)

| Knob                          | Default | Where it lives                | Notes                                              |
|-------------------------------|---------|-------------------------------|----------------------------------------------------|
| `detector`                    | `"orb"` | `pipeline.DEFAULTS`           | `"sift"` needs `opencv-contrib-python`             |
| `ratio` (Lowe's)              | 0.75    | `feature_matching`            | Smaller = stricter                                 |
| `spatial_distance_min` (px)   | 10.0    | `feature_matching`            | Drops trivially-close pairs                        |
| `ransac_threshold` (px)       | 3.0     | `geometric_verification`      | Reprojection error tolerance                       |
| `min_match_count`             | 10      | `geometric_verification`      | Hard floor before RANSAC runs                      |
| `max_iters`                   | 2000    | `geometric_verification`      | RANSAC iteration cap                               |
| `confidence`                  | 0.995   | `geometric_verification`      | RANSAC confidence                                  |
| `forgery_min_inliers`         | 8       | `pipeline.DEFAULTS`           | Inlier count floor for `forgery_detected=True`     |
| `forgery_min_inlier_ratio`    | 0.08    | `pipeline.DEFAULTS`           | inliers / total floor                              |

---

## 4. Integration Guide

### For Member B (UI / integration)

**Already wired.** `app/streamlit_app.py` calls
`run_copy_move_analysis(preprocess.gray_image, preprocess.color_image,
detector="orb")` and displays every field of `CopyMoveResult` in the
"Copy-Move" tab. See commit 9.

If you want a UI toggle between ORB and SIFT, just expose a checkbox and
pass it as the `detector` argument:

```python
detector = st.radio("Detector", ["orb", "sift"], horizontal=True)
copy_move = run_copy_move_analysis(preprocess.gray_image,
                                   preprocess.color_image,
                                   detector=detector)
```

The `CopyMoveResult` fields you can use directly:

```python
cm.forgery_detected          # bool
cm.num_total_matches         # int
cm.num_inlier_matches        # int
cm.confidence_score          # float in [0, 1]   (experimental)
cm.transform_params          # TransformParams or None
    .rotation_degrees        # float
    .scale_factor            # float (1.0 = no scale change)
    .translation_x           # float px
    .translation_y           # float px
    .translation_distance    # float px
cm.regions                   # list[RegionInfo]
cm.mask                      # uint8 HxW binary mask, or None
cm.match_visualization       # BGR HxW (matches drawn)  -- already drawn
cm.region_visualization      # BGR HxW (regions drawn)  -- already drawn
```

If you want to draw matches / regions yourself instead of using the
pre-rendered visualizations, the helpers are at:

```python
from src.copy_move.visualization import (
    draw_matches, draw_regions, draw_transformation_arrow,
)
```

### For Member C (tests / dataset / evaluation)

**Pipeline entry points for tests:**

```python
from src.copy_move.pipeline import run_copy_move_analysis, DEFAULTS
from src.copy_move.parameters import decompose_affine, reconstruct_affine
from src.utils.data_types import CopyMoveResult, TransformParams, RegionInfo
```

**Parameter-error evaluation** (matches `zed.md` §12):

```python
# For each forged image with logged (theta_true, scale_true, tx_true, ty_true):
res = run_copy_move_analysis(gray, color, detector="orb")
if res.transform_params is None:
    # missed detection — count as a false negative
    ...
tp = res.transform_params
theta_err   = abs(tp.rotation_degrees - theta_true)            # degrees
scale_err   = abs(tp.scale_factor - scale_true) / scale_true   # ratio
dist_err    = math.hypot(tp.translation_x - tx_true,
                         tp.translation_y - ty_true)            # px
```

**Sanity test before pushing pipeline changes:**

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path
python scripts/smoke_test_copy_move.py
```

This builds a synthetic forgery and checks that
`run_copy_move_analysis` recovers theta within 5°, scale within 10%,
and translation within 100 px. Exits 0 on pass.

**Known limitations to document in the final report:**

1. **Synthetic perfect pixel-copy fails.** The smoke test warps A
   before pasting B at the *unwarped* position — the inverse mapping is
   recoverable. If you instead paste a raw A and rotate *that* copy,
   the rotation appears in the recovered transform directly. The
   pipeline handles both consistently (just with the expected sign/direction).
2. **Highly symmetric objects** (checkerboards, brick walls) can cause
   RANSAC to lock onto a degenerate point-symmetric consensus (180°
   rotation) that has more inliers than the real forgery. Mitigation:
   use textured/asymmetric source content (real photos almost always
   work) or post-filter the consensus by checking that the recovered
   rotation lies within an expected range.
3. **Smooth / low-feature images** (sky, blank walls) produce few
   matches and may fail to detect. This is inherent to feature-based
   methods; a CNN-based detector would be the alternative.
4. **Confidence is inlier ratio**, NOT a calibrated probability. A
   value of 0.5 does NOT mean "50% chance of forgery". Label it
   "experimental" in every UI display (the Streamlit app already does
   this).
5. **Feature positions are in resized-image coordinates.** If you want
   to map the bbox / mask back to the original image dimensions, divide
   by `PreprocessResult.scale_factor`. The UI currently displays
   parameters in resized coords (which is the natural unit, since RANSAC
   ran on the resized image).

---

## 5. Phase 2 Integration Log (A + B → `develop`)

### Context discovered at merge time

- **Member B (RIJUL S, branch `feature/member-b-all`)** had already merged
  Member A's work into their branch at `614c254`, then built on top:
  - `f057a99` — completed Member B integration: enhanced Streamlit UI
    (344 lines), improved `ela.py`, `metadata.py`, `preprocessing.py`,
    added `scripts/smoke_test_full_pipeline.py`.
  - `67763d6` — committed `outputs/smoke_forged.png` as a test fixture
    (note: this re-introduces the `outputs/` dir locally; keep or drop
    per team agreement — Member B committed it deliberately for testing).
- Their branch already contained `main` (`5e5d8ff`), so integration was a
  fast-forward plus one small merge.

### Merge sequence (all on local `develop`)

| Step | Command | Result |
|---|---|---|
| 1 | `git checkout -b develop origin/main` | branched from `5e5d8ff` |
| 2 | `git merge origin/feature/member-b-all` | fast-forward → `67763d6` (needed local untracked `outputs/smoke_forged.png` removed first — it was a regenerated copy clashing with their tracked file) |
| 3 | `git merge feature/member-a-copy-move` | merge commit `826bed5`, conflict-free, only `.gitignore +4` (graphify local-only ignores) |

### Post-merge verification (all passed)

```
python scripts/smoke_test_copy_move.py     → PASS   (Member A: theta 0.01°, scale 0.08%)
python scripts/smoke_test_full_pipeline.py → PASS   (Member B: preprocess + copy-move + ELA + metadata + JSON export)
python -c "import app.streamlit_app"       → OK
```

### Branch state after integration

- `develop` = `826bed5` — contains ALL Member A + Member B work, verified.
- Local unique commits not on `origin`: `25dbcb3`, `9da9ba3` (gitignore only),
  and the merge `826bed5`. Nothing else — **not pushed yet.**
- `report.md` and `.gitattributes` remain untracked/gitignored (never commit).
- **Member C should branch from `develop`** (e.g. `feature/dataset`,
  `feature/testing`, `feature/evaluation` per zed.md §9).

### Still open

- [ ] Push `develop` (now at the Member C merge) to origin.
- [ ] PR `develop` → `main` (zed.md forbids direct commits to `main`).
- [x] Merge Member C's branch (`origin/feature/member-c-dataset`, `d567054`)
  into `develop` — dataset, pytest suite, evaluation scripts and docs are now
  on `develop`.

---

## 6. Phase 3 Hardening — zed.md compliance pass

### 6.1 Member A (copy-move)

| Change | File(s) | Why |
|---|---|---|
| Canonical match ordering + symmetric dedup | `feature_matching.py` | zed.md §Step 6.3. Each correspondence is kept once with `queryIdx < trainIdx`; the reverse pair carried the inverse transform and halved the inlier ratio / flipped rotation signs. |
| Repetitive lattice suppression | `feature_matching.py`, `pipeline.py` | zed.md §Step 6.4. Drops the dominant displacement group when ≥75% of matches share it and its source points span ≥60% of both axes. Targets brick/window false positives. |
| Physical centroid-to-centroid translation | `parameters.py`, `pipeline.py` | zed.md §Step 8. The affine translation column is origin-dependent under rotation/scale; `physical_translation()` reports the true paste displacement. |
| Adaptive multi-cluster localization | `localization.py` | zed.md §Steps 9–11. Replaces the greedy 2-cluster split with single-link clustering (supports k>2 copies, drops isolated points). |
| Native-resolution region metrics | `data_types.py`, `localization.py`, `pipeline.py` | `RegionInfo` now carries `*_native` fields; `pipeline(preprocess_scale=...)` fills them. |
| Tests | `tests/test_copy_move.py` | Canonical ordering, lattice suppression (keep/drop cases), physical translation, multi-copy clustering, native metrics. |

### 6.2 Member B (forensics / UI)

| Change | File(s) | Why |
|---|---|---|
| Statistical ELA | `ela.py` | zed.md §Step 12. Region-vs-background z-score (`z = |μ_region − μ_bg| / σ_bg`, high ≥ 3, medium ≥ 1.5) when a copy-move mask exists; otherwise an 8×8 block z-score. Detects lossless inputs and notes that ELA cannot reveal double-JPEG history there. |
| ELA wired to copy-move mask | `app/streamlit_app.py` | `run_full_pipeline` passes `copy_move.mask` into ELA. |
| Upload extension preservation | `app/streamlit_app.py` | zed.md §Step 15. `safe_upload_suffix()` keeps `.png`/`.jpg`/`.jpeg`; unknown falls back to `.jpg`. |
| Dual-scale reporting | `app/streamlit_app.py` | Native-photo translation and region metrics shown in the UI and JSON export. |
| EXIF orientation normalisation | `src/utils/image_io.py`, `preprocessing.py` | zed.md §Member B. Pillow + `ImageOps.exif_transpose` before resizing. |
| Planned utils modules | `src/utils/{image_io,validation,visualization}.py`, `src/forensics/__init__.py` | Files owned by Member B in zed.md were missing. |

### 6.3 Docs / repo

- `docs/methodology.md`, `docs/limitations.md` updated to the implemented
  algorithms (canonical ordering, lattice suppression, adaptive clustering,
  statistical ELA, native coordinates).
- `README.md`: clone URL and documentation links fixed (they pointed at a
  local `file:///c:/Users/bijo/...` path); test count no longer hardcoded.
- `LICENSE` (MIT) added.
- `outputs/` evaluation reports to be regenerated after the hardening pass
  (the pre-change numbers predate canonical deduplication, lattice
  suppression, and physical translation).



