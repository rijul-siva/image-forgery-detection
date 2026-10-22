"""
Tests for Member A's copy-move detection pipeline and submodules.
Verified and executed by Member C.
"""

import math
import numpy as np
import cv2
import pytest

from src.copy_move.feature_extraction import extract_features, FeatureSet
from src.copy_move.feature_matching import (
    match_features,
    suppress_repetitive_patterns,
    MatchPair,
)
from src.copy_move.displacement_clustering import (
    cluster_displacements,
    detect_periodic_families,
    harmonic_multiple,
    participation_ratio,
)
from src.copy_move.geometric_verification import (
    verify_geometric,
    verify_geometric_clustered,
)
from src.copy_move.parameters import (
    decompose_affine,
    reconstruct_affine,
    physical_translation,
)
from src.copy_move.localization import localize_forgery
from src.copy_move.pipeline import run_copy_move_analysis
from src.utils.data_types import TransformParams, CopyMoveResult


@pytest.fixture
def textured_patch():
    """Generates an asymmetric textured image patch rich in keypoints."""
    rng = np.random.default_rng(42)
    img = np.full((300, 300), 200, dtype=np.uint8)
    for _ in range(40):
        cx = int(rng.integers(20, 280))
        cy = int(rng.integers(20, 280))
        r = int(rng.integers(5, 15))
        cv2.circle(img, (cx, cy), r, int(rng.integers(10, 180)), -1)
    # Add non-symmetric lines
    cv2.rectangle(img, (30, 40), (80, 150), 30, -1)
    cv2.line(img, (20, 280), (280, 50), 0, 3)
    return img


@pytest.fixture
def blank_patch():
    """Generates a completely uniform 300x300 image with zero features."""
    return np.full((300, 300), 128, dtype=np.uint8)


# ---- 1. Feature Extraction Tests ----

def test_feature_extraction_orb(textured_patch):
    features = extract_features(textured_patch, detector="orb")
    assert isinstance(features, FeatureSet)
    assert features.detector_name == "orb"
    assert len(features.keypoints) > 10
    assert features.descriptors is not None
    assert features.descriptors.ndim == 2
    assert features.descriptors.shape[1] == 32
    assert features.descriptors.dtype == np.uint8


def test_feature_extraction_sift(textured_patch):
    features = extract_features(textured_patch, detector="sift")
    assert isinstance(features, FeatureSet)
    assert features.detector_name == "sift"
    assert len(features.keypoints) > 10
    assert features.descriptors is not None
    assert features.descriptors.shape[1] == 128
    assert features.descriptors.dtype == np.float32


def test_feature_extraction_blank(blank_patch):
    features = extract_features(blank_patch, detector="orb")
    assert len(features.keypoints) == 0
    assert features.descriptors is None or len(features.descriptors) == 0


def test_feature_extraction_invalid_inputs():
    with pytest.raises(ValueError, match="gray_image is None"):
        extract_features(None)
    with pytest.raises(ValueError, match="must be 2-D"):
        extract_features(np.zeros((100, 100, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Unknown detector"):
        extract_features(np.zeros((100, 100), dtype=np.uint8), detector="invalid_name")


# ---- 2. Feature Matching Tests ----

def test_match_features_spatial_suppression():
    """Verify that matches between keypoints that are too close spatially are rejected."""
    # Create synthetic descriptors and keypoints where distance between pt A and pt B is 5 px (< 10 px min)
    kp1 = cv2.KeyPoint(x=50.0, y=50.0, size=10)
    kp2 = cv2.KeyPoint(x=53.0, y=54.0, size=10)  # distance = 5.0 px
    kp3 = cv2.KeyPoint(x=200.0, y=200.0, size=10)

    # 3 identical descriptors
    desc = np.ones((3, 32), dtype=np.uint8)
    features = FeatureSet(keypoints=[kp1, kp2, kp3], descriptors=desc, detector_name="orb")

    # With spatial_distance_min = 10, kp1 and kp2 should not match each other
    matches = match_features(features, ratio=0.9, spatial_distance_min=10.0)
    for m in matches:
        dist = math.hypot(m.src_xy[0] - m.dst_xy[0], m.src_xy[1] - m.dst_xy[1])
        assert dist >= 10.0


def test_match_features_insufficient_descriptors():
    features = FeatureSet(keypoints=[], descriptors=np.zeros((2, 32), dtype=np.uint8), detector_name="orb")
    matches = match_features(features)
    assert matches == []


def test_match_features_canonical_ordering():
    """Every unordered correspondence must appear exactly once, with src_idx < dst_idx."""
    canvas = np.full((600, 800), 120, dtype=np.uint8)
    rng = np.random.default_rng(77)
    patch = np.full((120, 120), 220, dtype=np.uint8)
    for _ in range(25):
        cx, cy = int(rng.integers(10, 110)), int(rng.integers(10, 110))
        cv2.circle(patch, (cx, cy), int(rng.integers(4, 10)), int(rng.integers(10, 200)), -1)
    canvas[100:220, 100:220] = patch
    canvas[350:470, 500:620] = patch

    features = extract_features(canvas, detector="orb")
    matches = match_features(features, ratio=0.8, spatial_distance_min=10.0)

    assert len(matches) > 0
    seen = set()
    for m in matches:
        assert m.src_idx < m.dst_idx
        key = (m.src_idx, m.dst_idx)
        assert key not in seen
        seen.add(key)


def test_suppress_repetitive_patterns_keeps_localized_copy_move():
    """A copied object's source keypoints are localized, so matches survive."""
    matches = []
    for i in range(25):
        x = 100.0 + (i % 5) * 10.0
        y = 100.0 + (i // 5) * 10.0
        matches.append(MatchPair(
            src_idx=i, dst_idx=100 + i,
            src_xy=(x, y), dst_xy=(x + 300.0, y + 150.0), distance=10.0,
        ))
    out = suppress_repetitive_patterns(matches, image_shape=(600, 800))
    assert len(out) == len(matches)


def test_suppress_repetitive_patterns_drops_lattice():
    """Identical offsets with sources scattered across the image are periodic texture."""
    matches = []
    for row in range(4):
        for col in range(6):
            x = 50.0 + col * 130.0
            y = 50.0 + row * 130.0
            matches.append(MatchPair(
                src_idx=len(matches), dst_idx=1000 + len(matches),
                src_xy=(x, y), dst_xy=(x + 180.0, y + 40.0), distance=10.0,
            ))
    out = suppress_repetitive_patterns(matches, image_shape=(600, 800))
    assert out == []


# ---- 3. Geometric Verification (RANSAC) Tests ----

def test_verify_geometric_clean_transform():
    """Verify RANSAC recovers an exact similarity transformation from synthetic inliers."""
    # Ground truth: 20 deg rotation, scale 1.05, tx 80, ty -40
    theta = math.radians(20.0)
    s = 1.05
    tx, ty = 80.0, -40.0
    M_true = np.array([
        [s * math.cos(theta), -s * math.sin(theta), tx],
        [s * math.sin(theta),  s * math.cos(theta), ty]
    ], dtype=np.float64)

    rng = np.random.default_rng(123)
    matches = []
    # Generate 30 clean correspondences
    for i in range(30):
        src_pt = np.array([float(rng.integers(50, 300)), float(rng.integers(50, 300))])
        dst_pt = (M_true @ np.array([src_pt[0], src_pt[1], 1.0]))[:2]
        matches.append(MatchPair(
            src_idx=i, dst_idx=i + 30,
            src_xy=(float(src_pt[0]), float(src_pt[1])),
            dst_xy=(float(dst_pt[0]), float(dst_pt[1])),
            distance=10.0
        ))

    # Add 5 random outliers
    for j in range(5):
        matches.append(MatchPair(
            src_idx=100 + j, dst_idx=200 + j,
            src_xy=(float(rng.integers(10, 500)), float(rng.integers(10, 500))),
            dst_xy=(float(rng.integers(10, 500)), float(rng.integers(10, 500))),
            distance=50.0
        ))

    geo = verify_geometric(matches, ransac_threshold=3.0, min_match_count=10)
    assert geo.affine_matrix is not None
    assert geo.num_inliers >= 28  # Virtually all clean matches should be inliers
    assert geo.num_outliers >= 5

    # Check decomposed parameters
    tp = decompose_affine(geo.affine_matrix)
    assert tp.rotation_degrees == pytest.approx(20.0, abs=0.5)
    assert tp.scale_factor == pytest.approx(1.05, abs=0.02)
    assert tp.translation_x == pytest.approx(80.0, abs=1.0)
    assert tp.translation_y == pytest.approx(-40.0, abs=1.0)


def test_verify_geometric_pure_noise():
    """Verify pure random matches are rejected by RANSAC."""
    rng = np.random.default_rng(999)
    noise_matches = []
    for i in range(25):
        noise_matches.append(MatchPair(
            src_idx=i, dst_idx=i + 50,
            src_xy=(float(rng.uniform(0, 500)), float(rng.uniform(0, 500))),
            dst_xy=(float(rng.uniform(0, 500)), float(rng.uniform(0, 500))),
            distance=100.0
        ))
    geo = verify_geometric(noise_matches, ransac_threshold=2.0, min_match_count=10)
    # Pure random noise should either fail to form an affine model or yield very few inliers
    assert geo.affine_matrix is None or geo.num_inliers < 6


# ---- 4. Affine Decomposition & Reconstruction Tests ----

@pytest.mark.parametrize("angle,scale,tx,ty", [
    (0.0, 1.0, 100.0, 50.0),
    (35.5, 1.15, -60.0, 120.0),
    (-45.0, 0.85, 200.0, -150.0),
    (90.0, 1.0, 0.0, 0.0),
    (179.0, 1.02, 30.0, 40.0),
])
def test_decompose_and_reconstruct_affine_roundtrip(angle, scale, tx, ty):
    initial_params = TransformParams(
        rotation_degrees=float(angle),
        scale_factor=float(scale),
        translation_x=float(tx),
        translation_y=float(ty),
        translation_distance=math.hypot(tx, ty)
    )
    M = reconstruct_affine(initial_params)
    assert M.shape == (2, 3)

    recovered = decompose_affine(M)
    assert recovered.rotation_degrees == pytest.approx(angle, abs=1e-3)
    assert recovered.scale_factor == pytest.approx(scale, rel=1e-3)
    assert recovered.translation_x == pytest.approx(tx, abs=1e-3)
    assert recovered.translation_y == pytest.approx(ty, abs=1e-3)
    assert recovered.translation_distance == pytest.approx(initial_params.translation_distance, abs=1e-3)


def test_decompose_affine_edge_cases():
    assert decompose_affine(None) is None
    with pytest.raises(ValueError, match="Expected 2x3"):
        decompose_affine(np.zeros((3, 3)))


def test_physical_translation_is_origin_independent():
    """Physical paste displacement must not depend on the rotation origin."""
    theta = math.radians(30.0)
    s = 1.1
    offset = np.array([150.0, -60.0])
    rng = np.random.default_rng(5)
    src = np.column_stack([
        rng.uniform(50, 400, size=40),
        rng.uniform(50, 350, size=40),
    ])
    center = src.mean(axis=0)  # object centroid rotation center
    rot_scale = s * np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ])
    dst = (src - center) @ rot_scale.T + center + offset

    tx, ty, dist = physical_translation(src, dst)
    assert tx == pytest.approx(offset[0], abs=1e-6)
    assert ty == pytest.approx(offset[1], abs=1e-6)
    assert dist == pytest.approx(math.hypot(*offset), abs=1e-6)

    assert physical_translation(np.empty((0, 2)), np.empty((0, 2))) is None


# ---- 5. Localization Tests ----

def test_localization_two_clusters():
    """Verify localization clusters inliers, runs morphology, and computes region metrics."""
    # 10 points centered at (100, 100), 10 points centered at (400, 400)
    rng = np.random.default_rng(7)
    src_cluster = np.array([[100 + rng.uniform(-20, 20), 100 + rng.uniform(-20, 20)] for _ in range(12)])
    dst_cluster = np.array([[400 + rng.uniform(-20, 20), 400 + rng.uniform(-20, 20)] for _ in range(12)])

    loc = localize_forgery(src_cluster, dst_cluster, image_shape=(600, 800))
    assert loc.combined_mask.shape == (600, 800)
    assert len(loc.regions) in [1, 2]
    for r in loc.regions:
        assert r.info.area_px > 0
        assert r.info.bbox_width > 0
        assert r.info.bbox_height > 0
        assert 0 <= r.info.centroid_x <= 800
        assert 0 <= r.info.centroid_y <= 600


def test_localization_multi_copy_clusters():
    """Multi-copy forgeries produce more than two regions."""
    rng = np.random.default_rng(11)
    centers_src = [(100, 100), (500, 100), (900, 100)]
    centers_dst = [(100, 600), (500, 600), (900, 600)]

    def make_points(centers):
        return np.array([
            (cx + rng.uniform(-10, 10), cy + rng.uniform(-10, 10))
            for cx, cy in centers
            for _ in range(6)
        ])

    loc = localize_forgery(
        make_points(centers_src),
        make_points(centers_dst),
        image_shape=(1000, 1000),
    )
    assert len(loc.regions) == 6


def test_localization_native_coordinates():
    """Native metrics map back to original resolution via 1 / scale_factor."""
    src = np.array([[100 + i, 100 + i] for i in range(10)], dtype=float)
    dst = np.array([[400 + i, 300 + i] for i in range(10)], dtype=float)
    scale = 0.5  # analysis canvas is half the original size

    loc = localize_forgery(src, dst, image_shape=(600, 800), scale_factor=scale)
    assert len(loc.regions) >= 1
    for r in loc.regions:
        info = r.info
        assert info.bbox_x_native == pytest.approx(info.bbox_x * 2, abs=1)
        assert info.bbox_width_native == pytest.approx(info.bbox_width * 2, abs=1)
        assert info.area_px_native == pytest.approx(info.area_px * 4, rel=1e-6)
        assert info.centroid_x_native == pytest.approx(info.centroid_x * 2, rel=1e-6)
        assert info.centroid_y_native == pytest.approx(info.centroid_y * 2, rel=1e-6)


# ---- 6. End-to-End Copy-Move Pipeline Tests ----

def test_run_copy_move_analysis_blank_image(blank_patch):
    color = cv2.cvtColor(blank_patch, cv2.COLOR_GRAY2BGR)
    res = run_copy_move_analysis(blank_patch, color, detector="orb")
    assert isinstance(res, CopyMoveResult)
    assert res.forgery_detected is False
    assert res.num_inlier_matches == 0
    assert res.affine_matrix is None
    assert res.transform_params is None
    assert res.regions == []


def test_run_copy_move_analysis_synthetic_forgery():
    """Verify copy-move detects a copied and rotated object patch."""
    canvas = np.full((600, 800, 3), 120, dtype=np.uint8)

    # Make a distinctive patch
    rng = np.random.default_rng(21)
    patch = np.full((120, 120, 3), 220, dtype=np.uint8)
    for _ in range(25):
        cx, cy = int(rng.integers(10, 110)), int(rng.integers(10, 110))
        cv2.circle(patch, (cx, cy), int(rng.integers(4, 10)), (int(rng.integers(10, 200)), int(rng.integers(10, 200)), 20), -1)
    cv2.rectangle(patch, (20, 20), (60, 90), (0, 0, 180), -1)

    # Place source at (100, 100)
    canvas[100:220, 100:220] = patch
    # Place copy at (400, 250)
    canvas[250:370, 400:520] = patch

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    res = run_copy_move_analysis(
        gray, canvas, detector="orb",
        ratio=0.85, ransac_threshold=5.0,
        min_match_count=4, forgery_min_inliers=4
    )
    assert res.forgery_detected is True
    assert res.num_inlier_matches >= 4
    assert res.transform_params is not None
    assert res.transform_params.scale_factor == pytest.approx(1.0, abs=0.1)


# ---- 7. Displacement Clustering & Multi-Hypothesis Verification ----

def _lattice_matches(step=71.0, columns=8, rows=8, x0=100.0, y0=100.0, cell=12.0, max_hop=1):
    """
    Matches between the columns of a periodic grid (a weather table whose
    columns repeat every `step` pixels).

    The matches chain: the destination of one match is the source of the
    next one, which is exactly what makes a periodic texture different from
    a duplicated region. With `max_hop` > 1 the grid also produces the
    longer hops (column 1 -> 3), i.e. the harmonic structure of a lattice.
    """
    matches = []
    idx = 0
    for col in range(columns):
        for hop in range(1, max_hop + 1):
            if col + hop >= columns:
                continue
            for row in range(rows):
                for k in range(3):
                    x = x0 + col * step + k * 4.0
                    y = y0 + row * cell
                    matches.append(MatchPair(
                        src_idx=idx, dst_idx=10_000 + idx,
                        src_xy=(x, y), dst_xy=(x + step * hop, y), distance=10.0,
                    ))
                    idx += 1
    return matches


def _forgery_matches(count=40, dx=-224.4, dy=504.2, x0=900.0, y0=100.0, w=160.0, h=120.0):
    """Matches of a localized region that was copied elsewhere in the image."""
    rng = np.random.default_rng(5)
    matches = []
    for i in range(count):
        x = x0 + float(rng.uniform(0.0, w))
        y = y0 + float(rng.uniform(0.0, h))
        matches.append(MatchPair(
            src_idx=i, dst_idx=20_000 + i,
            src_xy=(x, y), dst_xy=(x + dx, y + dy), distance=10.0,
        ))
    return matches


def test_cluster_displacements_groups_by_shift_vector():
    """Matches with the same shift vector end up in the same cluster."""
    matches = _lattice_matches(step=71.0, columns=4, rows=4)
    clusters = cluster_displacements(matches, tolerance=3.0)

    assert len(clusters) == 1
    assert clusters[0].count == len(matches)
    assert clusters[0].displacement[0] == pytest.approx(71.0, abs=1.0)
    assert clusters[0].displacement[1] == pytest.approx(0.0, abs=1.0)


def test_cluster_displacements_separates_different_shifts():
    matches = _lattice_matches(step=71.0, columns=4, rows=4)
    matches += _forgery_matches(count=20)
    clusters = cluster_displacements(matches, tolerance=3.0)
    assert len(clusters) == 2
    assert sorted(c.count for c in clusters) == [20, len(matches) - 20]


def test_detect_periodic_families_finds_harmonic_lattice():
    """Shift vectors that are integer multiples of a base step form a lattice."""
    matches = []
    for n in range(1, 6):
        for i in range(12):
            matches.append(MatchPair(
                src_idx=len(matches), dst_idx=50_000 + len(matches),
                src_xy=(100.0 + i * 5.0, 200.0 + i * 3.0),
                dst_xy=(100.0 + i * 5.0 + 71.0 * n, 200.0 + i * 3.0),
                distance=10.0,
            ))

    clusters = cluster_displacements(matches, tolerance=3.0)
    families = detect_periodic_families(clusters)

    assert len(families) == 1
    family = families[0]
    assert family.base[0] == pytest.approx(71.0, abs=1.0)
    assert family.harmonics == [1, 2, 3, 4, 5]
    assert family.total_matches == len(matches)


def test_harmonic_multiple_rejects_non_parallel_vectors():
    """A vector that is not parallel to the base is not a harmonic of it."""
    base = np.array([71.0, 0.0])
    assert harmonic_multiple(base, np.array([142.0, 0.0])) == 2
    assert harmonic_multiple(base, np.array([-71.0, 0.0])) == -1
    # Tilted well away from the base axis.
    assert harmonic_multiple(base, np.array([200.0, 120.0])) is None
    # Parallel but far outside the base axis laterally.
    assert harmonic_multiple(base, np.array([142.0, 40.0])) is None


def test_harmonic_multiple_caps_harmonic_order():
    """A shift that is a large multiple of the base is not a lattice member."""
    base = np.array([5.0, 0.0])
    assert harmonic_multiple(base, np.array([35.0, 0.0]), max_multiple=16) == 7
    assert harmonic_multiple(base, np.array([100.0, 0.0]), max_multiple=16) is None


def test_participation_ratio_separates_lattice_from_duplicated_region():
    """A periodic texture chains; a one-off duplicated region does not."""
    lattice = _lattice_matches(step=71.0, columns=6, rows=6)
    lattice_src = np.array([m.src_xy for m in lattice])
    lattice_dst = np.array([m.dst_xy for m in lattice])
    assert participation_ratio(lattice_src, lattice_dst, tolerance=2.0) > 0.5

    forgery = _forgery_matches(count=40)
    forgery_src = np.array([m.src_xy for m in forgery])
    forgery_dst = np.array([m.dst_xy for m in forgery])
    assert participation_ratio(forgery_src, forgery_dst, tolerance=2.0) == 0.0


def test_verify_geometric_clustered_prefers_forgery_over_periodic_grid():
    """
    The single-model RANSAC trap: a periodic grid out-votes the forgery.

    The grid produces far more matches than the duplicated region, so a
    global RANSAC locks onto the grid. The clustered verification must
    still report the copy-move.
    """
    lattice = _lattice_matches(step=71.0, columns=12, rows=12)   # 396 matches
    forgery = _forgery_matches(count=40)
    matches = lattice + forgery

    global_result = verify_geometric(matches, ransac_threshold=3.0, min_match_count=10)
    assert global_result.num_inliers > 40  # the grid wins the majority vote

    result = verify_geometric_clustered(
        matches, image_shape=(900, 1200), ransac_threshold=3.0, min_match_count=10
    )

    assert result.affine_matrix is not None
    assert result.num_inliers >= 40
    # The winner is the duplicated region, not the grid.
    displacement = np.array([
        result.inlier_dst[:, 0].mean() - result.inlier_src[:, 0].mean(),
        result.inlier_dst[:, 1].mean() - result.inlier_src[:, 1].mean(),
    ])
    assert displacement[0] == pytest.approx(-224.4, abs=2.0)
    assert displacement[1] == pytest.approx(504.2, abs=2.0)


def test_verify_geometric_clustered_reports_periodic_lattice():
    """The periodic grid is recognised and reported instead of being a forgery."""
    lattice = _lattice_matches(step=71.0, columns=12, rows=12, max_hop=3)
    forgery = _forgery_matches(count=40)
    result = verify_geometric_clustered(
        lattice + forgery, image_shape=(900, 1200), ransac_threshold=3.0
    )

    assert len(result.periodic_families) >= 1
    summary = result.periodic_families[0].summary()
    assert summary["base_step_length"] == pytest.approx(71.0, abs=1.0)
    assert summary["harmonics"] == [1, 2, 3]

    # The winning hypothesis is not the lattice.
    assert result.candidate_scores[0]["periodicity"] < 0.5


def test_verify_geometric_clustered_pure_lattice_is_periodic():
    """Without a duplicated region the winner must look like periodic structure."""
    lattice = _lattice_matches(step=71.0, columns=12, rows=12)
    result = verify_geometric_clustered(
        lattice, image_shape=(900, 1200), ransac_threshold=3.0
    )

    assert result.candidate_scores, "expected at least one candidate hypothesis"
    assert result.candidate_scores[0]["periodicity"] > 0.5


def test_pipeline_ignores_periodic_ui_but_reports_forgery():
    """End-to-end: a screenshot with a periodic table plus a copied '22'."""
    canvas = np.full((900, 1200, 3), 245, dtype=np.uint8)

    # --- periodic weather table: identical columns every 71 px ---
    cell = np.full((90, 71, 3), 250, dtype=np.uint8)
    cv2.circle(cell, (30, 30), 20, (170, 170, 170), -1)
    cv2.circle(cell, (30, 30), 9, (90, 140, 200), -1)
    cv2.putText(cell, "22", (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
    for col in range(10):
        for row in range(6):
            canvas[120 + row * 90:210 + row * 90, 120 + col * 71:191 + col * 71] = cell

    # --- duplicated "22", scaled by 1.037 and pasted at the bottom left ---
    label = np.full((120, 160, 3), 250, dtype=np.uint8)
    cv2.putText(label, "22", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 6, cv2.LINE_AA)
    rng = np.random.default_rng(11)
    for _ in range(10):
        cv2.circle(
            label,
            (int(rng.integers(2, 155)), int(rng.integers(2, 115))),
            int(rng.integers(2, 7)),
            (int(rng.integers(60, 240)),) * 3, -1,
        )
    canvas[100:220, 880:1040] = label

    scaled = cv2.resize(label, None, fx=1.037, fy=1.037, interpolation=cv2.INTER_LINEAR)
    sh, sw = scaled.shape[:2]
    dst_x, dst_y = 880 - 224, 100 + 504
    canvas[dst_y:dst_y + sh, dst_x:dst_x + sw] = scaled

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    res = run_copy_move_analysis(gray, canvas, detector="orb", ratio=0.75)

    assert res.forgery_detected is True
    assert res.transform_params is not None
    tp = res.transform_params
    # The two regions are identical, so which one is the "copy" is not
    # recoverable from the geometry — accept either orientation.
    assert tp.scale_factor == pytest.approx(1.037, abs=0.05) or \
        tp.scale_factor == pytest.approx(1.0 / 1.037, abs=0.05)
    assert tp.translation_x * tp.translation_y < 0  # diagonal, opposite signs
    assert math.hypot(tp.translation_x, tp.translation_y) == pytest.approx(
        math.hypot(224.0, 504.0), abs=30.0
    )
    # The periodic table is recognised, not reported as the forgery.
    assert any(
        abs(lattice["base_step_length"] - 71.0) < 3.0
        for lattice in res.periodic_lattices
    )
