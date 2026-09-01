import cv2
import numpy as np
from src.utils.data_types import ELAResult

def run_ela_analysis(
    image_path: str,
    quality: int = 95,
    amplification: int = 20
) -> ELAResult:
    """
    Perform Error Level Analysis (ELA) on an image.
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image at path: {image_path}")
        
    # Compress the image
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', original, encode_param)
    compressed = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    
    # Calculate absolute difference
    diff = cv2.absdiff(original, compressed)
    
    # Amplify the differences
    ela_image = diff * amplification
    # Clip values to 255 to avoid overflow artifacts
    ela_image = np.clip(ela_image, 0, 255).astype(np.uint8)
    
    # Very basic suspicion level logic (just as a placeholder)
    max_diff = np.max(ela_image)
    if max_diff > 200:
        suspicion_level = "high"
    elif max_diff > 100:
        suspicion_level = "medium"
    else:
        suspicion_level = "low"
        
    return ELAResult(
        ela_image=ela_image,
        suspicion_level=suspicion_level,
        description=f"ELA analysis complete (Quality: {quality}). Max difference: {max_diff}."
    )
