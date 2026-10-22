"""
Displacement-vector clustering for copy-move forgery detection.

Why this module exists
----------------------
`cv2.estimateAffinePartial2D` searches for ONE dominant similarity model in
the whole match set. A screenshot that contains a periodic UI (a weather
table whose columns repeat every ~71 px, a row of identical widgets, a tiled
grid) produces hundreds of matches that share a single translation. That
periodic family can easily out-vote a genuine, localized copy-move forgery,
so the single global fit locks onto the lattice and the forgery is reported
as a pile of red outliers.

Classical copy-move literature (Amerini et al., "Image Forgery Detection by
Duplicating the Same Image Region", IEEE TIFS 2012) groups feature matches by
their spatial shift vector

    d = (x2 - x1, y2 - y1)

before fitting any model, so that competing explanations can be evaluated
separately instead of by majority vote.

This module provides the building blocks:

  1. `cluster_displacements`   — group matches whose shift vectors agree.
  2. `detect_periodic_families` — recognise shift vectors that are integer
     multiples of a base step (a multi-harmonic lattice) and report them as
     periodic structure rather than as a forgery.
  3. `participation_ratio`     — measure how strongly a set of
     correspondences *chains* (each destination region is itself the source
     of another match). Periodic textures chain; a one-off duplicated region
     does not.

Nothing here depends on anything outside `src/copy_move/`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .feature_matching import MatchPair


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DisplacementCluster:
    """A group of matches whose shift vectors agree within a tolerance."""

    match_indices: List[int]
    displacement: np.ndarray  # (2,) mean shift vector

    @property
    def count(self) -> int:
        return len(self.match_indices)

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.displacement[0], self.displacement[1]))


@dataclass
class PeriodicFamily:
    """
    A set of displacement clusters whose shift vectors are integer multiples
    of a common base step, e.g. (+71, 0), (+142, 0), (+213, 0), ...

    Such a family is the signature of a periodic structure (repeated table
    columns, a tiled grid, a row of identical widgets) rather than of a
    duplicated region.

    The multiples are signed: a lattice axis is undirected, so the family
    also contains the matches that run the other way (-71, -142, ...).
    """

    base: np.ndarray                                   # (2,) base step
    members: List[Tuple[int, DisplacementCluster]]     # (multiple n, cluster)

    @property
    def total_matches(self) -> int:
        return sum(c.count for _, c in self.members)

    @property
    def multiples(self) -> List[int]:
        return sorted({int(n) for n, _ in self.members})

    @property
    def harmonics(self) -> List[int]:
        """Distinct harmonic orders of the family (magnitudes, ascending)."""
        return sorted({abs(int(n)) for n, _ in self.members})

    @property
    def match_indices(self) -> List[int]:
        out: List[int] = []
        for _, cluster in self.members:
            out.extend(cluster.match_indices)
        return out

    def summary(self) -> dict:
        """Plain-data summary (handy for the UI / JSON export)."""
        return {
            "base_step_x": float(self.base[0]),
            "base_step_y": float(self.base[1]),
            "base_step_length": float(np.hypot(self.base[0], self.base[1])),
            "multiples": self.multiples,
            "harmonics": self.harmonics,
            "num_matches": int(self.total_matches),
        }


# ---------------------------------------------------------------------------
# Shift vectors
# ---------------------------------------------------------------------------

def displacement_vectors(matches: Sequence[MatchPair]) -> np.ndarray:
    """(N, 2) array of shift vectors d = (x2 - x1, y2 - y1)."""
    if not matches:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(
        [
            [m.dst_xy[0] - m.src_xy[0], m.dst_xy[1] - m.src_xy[1]]
            for m in matches
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _leader_split(
    disp: np.ndarray,
    members: Sequence[int],
    tolerance: float,
) -> List[DisplacementCluster]:
    """
    Split a set of indices into compact clusters.

    Each cluster is seeded by the member with the most neighbours inside the
    tolerance ("leader"); every remaining member within tolerance of the
    leader joins it. Whatever is left over is repeated with the next leader.
    This keeps clusters compact instead of letting them chain through
    intermediate points.
    """
    clusters: List[DisplacementCluster] = []
    remaining = list(members)
    while remaining:
        arr = np.asarray(remaining, dtype=np.int64)
        pts = disp[arr]
        # pairwise distances inside the candidate group
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        support = (dist <= tolerance).sum(axis=1)
        leader = int(np.argmax(support))
        base = pts[leader]
        keep_mask = np.linalg.norm(pts - base, axis=1) <= tolerance
        kept = arr[keep_mask]
        clusters.append(
            DisplacementCluster(
                match_indices=[int(i) for i in kept],
                displacement=disp[kept].mean(axis=0),
            )
        )
        keep_set = set(int(i) for i in kept)
        remaining = [i for i in remaining if i not in keep_set]
    return clusters


def cluster_displacements(
    matches: Sequence[MatchPair],
    tolerance: float = 3.0,
) -> List[DisplacementCluster]:
    """
    Group matches by their shift vector.

    Parameters
    ----------
    matches : sequence[MatchPair]
        Candidate matches (output of `match_features`).
    tolerance : float
        Maximum Euclidean distance (pixels) between two shift vectors for
        them to be considered the same displacement.

    Returns
    -------
    list[DisplacementCluster]
        Clusters sorted by descending size.
    """
    disp = displacement_vectors(matches)
    n = len(disp)
    if n == 0:
        return []
    tol = max(float(tolerance), 1e-6)

    # --- bucket by a grid of cell size `tolerance` so that neighbours can be
    # --- found without an O(n^2) all-pairs scan.
    cells: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n):
        key = (
            int(math.floor(disp[i, 0] / tol)),
            int(math.floor(disp[i, 1] / tol)),
        )
        cells.setdefault(key, []).append(i)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (cx, cy), idxs in cells.items():
        neighbours: List[int] = []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                neighbours.extend(cells.get((cx + ox, cy + oy), ()))
        for i in idxs:
            for j in neighbours:
                if j <= i:
                    continue
                if math.hypot(disp[i, 0] - disp[j, 0], disp[i, 1] - disp[j, 1]) <= tol:
                    union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters: List[DisplacementCluster] = []
    for members in groups.values():
        clusters.extend(_leader_split(disp, members, tol))

    clusters.sort(key=lambda c: (-c.count, c.displacement[0], c.displacement[1]))
    return clusters


# ---------------------------------------------------------------------------
# Periodic lattice detection
# ---------------------------------------------------------------------------

def harmonic_multiple(
    base: np.ndarray,
    displacement: np.ndarray,
    angle_tol_deg: float = 3.0,
    relative_tol: float = 0.06,
    absolute_tol: float = 4.0,
    max_multiple: int = 16,
) -> Optional[int]:
    """
    Return `n` if `displacement` is (approximately) the integer multiple
    `n * base`, otherwise None.

    The vector is decomposed in the frame of the base step:

        parallel = <displacement, base/|base|>
        lateral  = <displacement, perpendicular(base)>

    A genuine lattice keeps the lateral component at zero (every column is
    shifted along the same axis), so the lateral residual is bounded by a
    few pixels — *not* by a percentage of the total length. A percentage
    bound would happily absorb a long, slightly tilted vector (e.g. a
    genuine forgery offset that happens to point roughly along the lattice
    axis) into the family, which would then silence the very forgery we are
    looking for.

    `max_multiple` caps the harmonic order: the repeats of a periodic
    structure are *neighbours* (column 1->2, 2->3, ...), so a shift of 70
    base steps is a coincidence, not a lattice member. Without the cap any
    displacement that happens to be a multiple of a small step (a copy
    moved by 350 px in a patch that repeats every 5 px) would be mistaken
    for periodic structure.
    """
    base_len = float(np.hypot(base[0], base[1]))
    disp_len = float(np.hypot(displacement[0], displacement[1]))
    if base_len < 1e-9 or disp_len < 1e-9:
        return None
    if disp_len < base_len - absolute_tol:
        return None

    ux, uy = base[0] / base_len, base[1] / base_len
    parallel = displacement[0] * ux + displacement[1] * uy
    lateral = -displacement[0] * uy + displacement[1] * ux

    # Must be (nearly) parallel to the base axis.
    if abs(lateral) > max(absolute_tol, 0.06 * base_len):
        return None
    if abs(math.degrees(math.atan2(lateral, abs(parallel)))) > angle_tol_deg:
        return None

    n = int(round(parallel / base_len))
    if n == 0 or abs(n) > max_multiple:
        return None

    expected = n * base_len
    if abs(parallel - expected) > max(absolute_tol, relative_tol * abs(expected)):
        return None
    return n


def _axis_key(base: np.ndarray) -> Tuple[float, float]:
    """Canonical (sign-independent) direction of a lattice axis."""
    length = float(np.hypot(base[0], base[1]))
    if length < 1e-9:
        return (0.0, 0.0)
    ux, uy = base[0] / length, base[1] / length
    if ux < 0 or (abs(ux) < 1e-9 and uy < 0):
        ux, uy = -ux, -uy
    return (round(ux, 3), round(uy, 3))


def detect_periodic_families(
    clusters: Sequence[DisplacementCluster],
    min_multiples: int = 2,
    min_total_matches: int = 20,
    min_base_length: float = 8.0,
    angle_tol_deg: float = 3.0,
    relative_tol: float = 0.06,
    absolute_tol: float = 4.0,
    max_multiple: int = 16,
) -> List[PeriodicFamily]:
    """
    Find groups of displacement clusters that form a multi-harmonic lattice.

    A lattice shows up as several shift vectors that are integer multiples of
    one base step (+71, +142, +213, ... for columns 71 px apart). Requiring
    at least `min_multiples` distinct harmonics keeps ordinary
    single-offset repetition out of this detector — those are handled by the
    scatter heuristic in `feature_matching.suppress_repetitive_patterns`.

    The axis is undirected: the matches that run the other way (-71, -142,
    ...) belong to the same lattice and are collected into the same family.

    The harmonic order is capped by `max_multiple`: the repeats of a
    periodic structure are *neighbours* (column 1->2, 2->3, ...), so a
    shift of 70 base steps is a coincidence, not a lattice member. Without
    the cap, a copy that was moved by exactly 70 base steps would be
    mistaken for periodic structure and the forgery would be silenced.

    Returns
    -------
    list[PeriodicFamily]
        Families sorted by descending support, one per lattice axis.
    """
    by_axis: Dict[Tuple[float, float], PeriodicFamily] = {}

    for base_cluster in clusters:
        base = base_cluster.displacement
        if float(np.hypot(base[0], base[1])) < min_base_length:
            continue

        members: List[Tuple[int, DisplacementCluster]] = []
        for cluster in clusters:
            n = harmonic_multiple(
                base,
                cluster.displacement,
                angle_tol_deg=angle_tol_deg,
                relative_tol=relative_tol,
                absolute_tol=absolute_tol,
                max_multiple=max_multiple,
            )
            if n is not None:
                members.append((n, cluster))

        if len({abs(n) for n, _ in members}) < min_multiples:
            continue
        total = sum(cluster.count for _, cluster in members)
        if total < min_total_matches:
            continue

        axis = _axis_key(base)
        existing = by_axis.get(axis)
        if existing is None or total > existing.total_matches:
            by_axis[axis] = PeriodicFamily(base=base, members=members)
        else:
            # Same axis, weaker family: keep the stronger one's base but
            # remember the extra members.
            known = {id(cluster) for _, cluster in existing.members}
            for n, cluster in members:
                if id(cluster) not in known:
                    existing.members.append((n, cluster))
                    known.add(id(cluster))

    families = list(by_axis.values())
    families.sort(key=lambda f: (-f.total_matches, float(np.hypot(f.base[0], f.base[1]))))
    return families


# ---------------------------------------------------------------------------
# Chaining (periodic texture) evidence
# ---------------------------------------------------------------------------

def participation_ratio(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    tolerance: float = 6.0,
) -> float:
    """
    Fraction of destination points that are *also* the source of another
    correspondence.

    In a periodic texture every cell is a copy of its neighbour, so the
    region that receives a match also sends matches of its own — the
    correspondences chain (FRI->SAT->SUN->...). A duplicated (forged)
    region is a dead end: the pasted copy is not itself duplicated anywhere
    else, so nothing chains.

    Returns a value in [0, 1]; 0 means "no chaining at all".
    """
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 2)
    if len(src) == 0 or len(src) != len(dst):
        return 0.0

    hits = 0
    chunk = 512
    for start in range(0, len(dst), chunk):
        block = dst[start:start + chunk]
        dist = np.linalg.norm(src[None, :, :] - block[:, None, :], axis=2)
        for row in range(dist.shape[0]):
            dist[row, start + row] = np.inf  # never count a match against itself
        hits += int(np.count_nonzero(dist.min(axis=1) <= tolerance))
    return float(hits) / float(len(dst))


def spatial_spread(
    pts: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """
    Fraction of the image width / height covered by the 5th-95th percentile
    extent of `pts`.
    """
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(arr) == 0:
        return 0.0, 0.0
    if image_shape is not None:
        h, w = float(image_shape[0]), float(image_shape[1])
    else:
        span = np.ptp(arr, axis=0)
        w, h = max(float(span[0]), 1.0), max(float(span[1]), 1.0)
    x_lo, x_hi = np.percentile(arr[:, 0], [5, 95])
    y_lo, y_hi = np.percentile(arr[:, 1], [5, 95])
    return float(x_hi - x_lo) / w, float(y_hi - y_lo) / h


def _near_any(
    query: np.ndarray,
    reference: np.ndarray,
    tolerance: float,
) -> bool:
    """True when `query` lies within `tolerance` of any point in `reference`."""
    if len(reference) == 0:
        return False
    cell = max(float(tolerance), 1e-6)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, pt in enumerate(reference):
        key = (int(math.floor(pt[0] / cell)), int(math.floor(pt[1] / cell)))
        buckets.setdefault(key, []).append(i)

    qx, qy = float(query[0]), float(query[1])
    cx, cy = int(math.floor(qx / cell)), int(math.floor(qy / cell))
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for idx in buckets.get((cx + ox, cy + oy), ()):
                if math.hypot(reference[idx, 0] - qx, reference[idx, 1] - qy) <= tolerance:
                    return True
    return False


def chain_evidence_ratio(
    matches: Sequence[MatchPair],
    inlier_indices: Sequence[int],
    tolerance: float = 8.0,
    magnitude_band: Tuple[float, float] = (0.3, 3.3),
) -> float:
    """
    How strongly a hypothesis is embedded in a repeating structure.

    The correspondences of a genuine copy-move form a single relation
    between two regions: the source and its pasted copy. The correspondences
    of a periodic structure form many relations — a column matched to the
    next column *and* to the column after that.

    This function counts the matches that

      * touch one of the hypothesis' regions,
      * describe a *different* relation (a different shift vector), and
      * have a shift magnitude comparable to the hypothesis' own.

    The comparable-magnitude requirement matters: a duplicated patch often
    contains fine internal detail that repeats at a much smaller scale (a
    grid of identical icons, for instance). That internal repetition is not
    evidence that the region is embedded in a global lattice, so relations
    an order of magnitude smaller are ignored.

    The reverse orientation of the hypothesis' own relation is ignored as
    well — it describes the same pair of regions.

    Returns a value in [0, 1]; 0 means "this region is a dead end", i.e.
    nothing else in the image repeats it.
    """
    indices = list(inlier_indices)
    if not indices or not matches:
        return 0.0

    src = np.array([matches[i].src_xy for i in indices], dtype=np.float64)
    dst = np.array([matches[i].dst_xy for i in indices], dtype=np.float64)
    reference = np.vstack([src, dst])

    displacement = dst - src
    hyp_disp = np.median(displacement, axis=0)
    hyp_mag = float(np.hypot(hyp_disp[0], hyp_disp[1]))
    if hyp_mag < 1e-6:
        return 0.0
    disp_tol = max(tolerance, 0.03 * hyp_mag)

    low, high = magnitude_band
    hypothesis_set = set(indices)
    evidence = 0
    for i, match in enumerate(matches):
        if i in hypothesis_set:
            continue
        d = np.array(
            [match.dst_xy[0] - match.src_xy[0], match.dst_xy[1] - match.src_xy[1]],
            dtype=np.float64,
        )
        if np.hypot(d[0] - hyp_disp[0], d[1] - hyp_disp[1]) <= disp_tol:
            continue  # the same relation, same orientation
        if np.hypot(d[0] + hyp_disp[0], d[1] + hyp_disp[1]) <= disp_tol:
            continue  # the same relation, opposite orientation
        magnitude = float(np.hypot(d[0], d[1]))
        if not (low * hyp_mag <= magnitude <= high * hyp_mag):
            continue  # a different scale of repetition (internal detail)
        if _near_any(np.asarray(match.src_xy, dtype=np.float64), reference, tolerance) or \
                _near_any(np.asarray(match.dst_xy, dtype=np.float64), reference, tolerance):
            evidence += 1

    return min(1.0, evidence / float(len(indices)))


__all__ = [
    "DisplacementCluster",
    "PeriodicFamily",
    "chain_evidence_ratio",
    "cluster_displacements",
    "detect_periodic_families",
    "displacement_vectors",
    "harmonic_multiple",
    "participation_ratio",
    "spatial_spread",
]
