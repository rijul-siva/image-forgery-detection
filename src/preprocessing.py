import cv2
import numpy as np
from src.utils.data_types import PreprocessResult

def preprocess_image(image_path: str, max_edge: int = 1600) -> PreprocessResult:
    """
    Load, validate, resize, convert to grayscale, and optionally denoise.
    Returns a PreprocessResult for all downstream modules.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    # Validate dimensions
    original_height, original_width = image.shape[:2]

    # Calculate resize scale factor
    scale_factor = 1.0
    if max(original_height, original_width) > max_edge:
        if original_width > original_height:
            scale_factor = max_edge / original_width
        else:
            scale_factor = max_edge / original_height
            
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return PreprocessResult(
        color_image=image,
        gray_image=gray_image,
        scale_factor=scale_factor,
        original_width=original_width,
        original_height=original_height
    )
