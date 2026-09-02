"""
Decomposition of the 2x3 affine (similarity) matrix estimated by RANSAC
into rotation angle, uniform scale factor, and translation vector.

Similarity transform (2x3):
    M = [ a  b  tx ]
        [ c  d  ty ]

Properties of a similarity transform:
    a = s * cos(theta)
    b = -s * sin(theta)
    c = s * sin(theta)
    d = s * cos(theta)

So:
    s = sqrt(a^2 + c^2)
    theta = atan2(c, a)    (OpenCV's atan2 convention; result in radians)
    tx, ty = matrix translation column
    translation_distance = sqrt(tx^2 + ty^2)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.utils.data_types import TransformParams


def decompose_affine(affine_matrix: Optional[np.ndarray]) -> Optional[TransformParams]:
    """
    Convert a 2x3 affine matrix into a `TransformParams` dataclass.

    Returns None if the matrix is None or has the wrong shape.
    The returned `TransformParams.rotation_degrees` is in degrees, in the
    range (-180, 180].
    """
    if affine_matrix is None:
        return None
    if affine_matrix.shape != (2, 3):
        raise ValueError(
            f"Expected 2x3 affine matrix, got shape {affine_matrix.shape}"
        )

    a, b, tx = float(affine_matrix[0, 0]), float(affine_matrix[0, 1]), float(affine_matrix[0, 2])
    c, d, ty = float(affine_matrix[1, 0]), float(affine_matrix[1, 1]), float(affine_matrix[1, 2])

    scale = math.sqrt(a * a + c * c)
    # atan2(c, a) is the standard decomposition; alt is atan2(-b, d).
    theta_rad = math.atan2(c, a)
    theta_deg = math.degrees(theta_rad)
    # Wrap to (-180, 180]
    theta_deg = ((theta_deg + 180.0) % 360.0) - 180.0

    translation_distance = math.sqrt(tx * tx + ty * ty)

    return TransformParams(
        rotation_degrees=float(theta_deg),
        scale_factor=float(scale),
        translation_x=float(tx),
        translation_y=float(ty),
        translation_distance=float(translation_distance),
    )


def reconstruct_affine(params: TransformParams) -> np.ndarray:
    """
    Inverse of `decompose_affine`. Useful for unit-testing the decomposition
    and for downstream consumers that need a callable transform matrix.
    """
    theta = math.radians(params.rotation_degrees)
    s = params.scale_factor
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    M = np.array(
        [
            [s * cos_t, -s * sin_t, params.translation_x],
            [s * sin_t,  s * cos_t, params.translation_y],
        ],
        dtype=np.float64,
    )
    return M
