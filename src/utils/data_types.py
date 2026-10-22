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
    denoised_image: Optional[np.ndarray] = None  # Gaussian smoothed image (DIP Stage 1)


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
    """Metrics for one suspicious region.

    The primary fields are measured on the resized analysis canvas. The
    `*_native` fields are the same measurements mapped back to the original
    smartphone-photo resolution (requires `scale_factor` from preprocessing;
    they fall back to the resized values when it is 1.0).
    """
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    area_px: float                   # Area in pixels squared
    area_percent: float              # Area as % of total image area
    centroid_x: float
    centroid_y: float
    bbox_x_native: Optional[int] = None
    bbox_y_native: Optional[int] = None
    bbox_width_native: Optional[int] = None
    bbox_height_native: Optional[int] = None
    area_px_native: Optional[float] = None
    centroid_x_native: Optional[float] = None
    centroid_y_native: Optional[float] = None


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
    # Intermediate DIP stage outputs (for step-by-step forensic inspection)
    keypoints_visualization: Optional[np.ndarray] = None  # Scale-space keypoints (DIP Stage 2)
    raw_matches_visualization: Optional[np.ndarray] = None  # Filtered candidate matches before RANSAC (DIP Stage 3)
    raw_mask: Optional[np.ndarray] = None  # Binary keypoint footprint mask before morphology (DIP Stage 5.1)
    closed_mask: Optional[np.ndarray] = None  # Mask after morphological closing (DIP Stage 5.2)
    dilated_mask: Optional[np.ndarray] = None  # Mask after morphological dilation (DIP Stage 5.3)
    # Displacement-clustering diagnostics: periodic lattices (repeating UI
    # structure) that were recognised and excluded from the forgery verdict.
    periodic_lattices: list = field(default_factory=list)      # list of dicts (base step, multiples, match count)
    num_displacement_clusters: int = 0                         # number of distinct shift-vector groups
    candidate_scores: list = field(default_factory=list)       # per-hypothesis score breakdown


@dataclass
class ELAResult:
    """Output of Error Level Analysis."""
    ela_image: np.ndarray            # ELA heatmap visualization
    suspicion_level: str             # "low", "medium", "high"
    description: str                 # Human-readable summary
    # Intermediate DIP stage outputs
    compressed_image: Optional[np.ndarray] = None  # Re-compressed JPEG (DIP Stage 6.1)
    raw_diff: Optional[np.ndarray] = None  # Absolute pixel difference |I - I_JPEG| (DIP Stage 6.2)
    colormap_image: Optional[np.ndarray] = None  # Pseudocolor Jet heatmap (DIP Stage 6.4)
    mean_diff: float = 0.0  # Mean absolute error
    max_diff: float = 0.0  # Max absolute error


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
