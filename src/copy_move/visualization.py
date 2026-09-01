"""
Visualization helpers for the copy-move pipeline.

Kept inside `src/copy_move/` (not in `src/utils/visualization.py`) so that
Member A's package has zero cross-team runtime dependencies. Member B's UI
can import from here, or simply use the pre-rendered visualizations already
attached to `CopyMoveResult.match_visualization` and `region_visualization`.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .feature_matching import MatchPair
from .localization import LocalizedRegion


# Colors (BGR).
_INLIER_COLOR = (0, 255, 0)      # green for inlier matches
_OUTLIER_COLOR = (0, 0, 255)     # red for outlier / rejected matches
_REGION_BOX_COLOR = (0, 165, 255)  # orange bounding boxes
_REGION_OVERLAY_COLOR = (0, 165, 255)


def draw_matches(
    color_image: np.ndarray,
    matches: List[MatchPair],
    inlier_mask: Optional[np.ndarray] = None,
    max_draw: int = 200,
) -> np.ndarray:
    """
    Draw match lines on the color image.

    Parameters
    ----------
    color_image : np.ndarray
        BGR image (e.g. output of preprocessing).
    matches : list[MatchPair]
        Candidate matches.
    inlier_mask : np.ndarray | None
        Optional bool array marking which matches are inliers (green) vs
        outliers (red). If None, all are drawn green.
    max_draw : int
        Cap on matches drawn to keep the visualization readable.

    Returns
    -------
    np.ndarray
        BGR image copy with match lines drawn.
    """
    if color_image is None or not matches:
        return color_image.copy() if color_image is not None else np.zeros((1, 1, 3), dtype=np.uint8)

    out = color_image.copy()
    n = len(matches)
    if n > max_draw:
        # Draw evenly-spaced subset to stay readable.
        step = max(1, n // max_draw)
        idxs = list(range(0, n, step))[:max_draw]
    else:
        idxs = list(range(n))

    for i in idxs:
        m = matches[i]
        is_inlier = bool(inlier_mask[i]) if inlier_mask is not None else True
        color = _INLIER_COLOR if is_inlier else _OUTLIER_COLOR
        pt1 = (int(round(m.src_xy[0])), int(round(m.src_xy[1])))
        pt2 = (int(round(m.dst_xy[0])), int(round(m.dst_xy[1])))
        cv2.line(out, pt1, pt2, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(out, pt1, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, pt2, 3, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_regions(
    color_image: np.ndarray,
    regions: List[LocalizedRegion],
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Overlay each region's mask as a semi-transparent fill and draw its
    bounding box + centroid.
    """
    if color_image is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    out = color_image.copy()
    overlay = out.copy()
    for region in regions:
        m = region.mask
        if m is None or m.size == 0:
            continue
        # Resize mask if shape mismatch (safety).
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros_like(out)
        color_mask[m > 0] = _REGION_OVERLAY_COLOR
        overlay = np.where(color_mask > 0, color_mask, overlay)

    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    for region in regions:
        info = region.info
        x, y, w, h = info.bbox_x, info.bbox_y, info.bbox_width, info.bbox_height
        cv2.rectangle(out, (x, y), (x + w, y + h), _REGION_BOX_COLOR, 2)
        cx, cy = int(info.centroid_x), int(info.centroid_y)
        cv2.circle(out, (cx, cy), 5, _REGION_BOX_COLOR, -1)
        cv2.putText(
            out,
            f"{info.area_percent:.1f}%",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _REGION_BOX_COLOR,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_transformation_arrow(
    color_image: np.ndarray,
    regions: List[LocalizedRegion],
) -> np.ndarray:
    """
    If at least 2 regions exist, draw an arrow from the larger centroid
    to the smaller one — visualises the translation vector.
    """
    if color_image is None or len(regions) < 2:
        return color_image.copy() if color_image is not None else np.zeros((1, 1, 3), dtype=np.uint8)

    out = color_image.copy()
    a = regions[0].info
    b = regions[1].info
    p1 = (int(a.centroid_x), int(a.centroid_y))
    p2 = (int(b.centroid_x), int(b.centroid_y))
    cv2.arrowedLine(out, p1, p2, (255, 255, 255), 2, tipLength=0.05)
    return out
