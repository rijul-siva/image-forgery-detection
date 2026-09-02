"""
EXIF Metadata Extraction & Analysis module — owned by Member B.
Extracts camera, timestamp, software tags, and flags suspicious metadata tampering.
"""

from __future__ import annotations
import os
from PIL import Image
from PIL.ExifTags import TAGS
from src.utils.data_types import MetadataResult

SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "canva", "snapseed", 
    "pixlr", "paint.net", "affinity", "illustrator", "editor"
]

def _make_serializable(val):
    """Recursively convert EXIF values into JSON-friendly primitives."""
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    elif isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="ignore").strip("\x00")
        except Exception:
            return str(val)
    elif isinstance(val, (tuple, list)):
        return [_make_serializable(x) for x in val]
    elif isinstance(val, dict):
        return {str(k): _make_serializable(v) for k, v in val.items()}
    else:
        return str(val)

def run_metadata_analysis(image_path: str) -> MetadataResult:
    """
    Extract and analyze EXIF/image metadata.

    Parameters
    ----------
    image_path : str
        Path to input image file.

    Returns
    -------
    MetadataResult
        Dataclass containing metadata dict, editing software flag, software name, 
        metadata_stripped boolean, and suspicious flags list.
    """
    metadata_dict = {}
    has_editing_software = False
    editing_software = None
    flags = []

    try:
        with Image.open(image_path) as img:
            # Extract EXIF tags via standard Pillow getexif()
            exifdata = img.getexif()
            
            if exifdata:
                for tag_id, data in exifdata.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    metadata_dict[str(tag_name)] = _make_serializable(data)

            # Extra info from Image properties (format, mode, size)
            metadata_dict["Format"] = str(img.format)
            metadata_dict["Mode"] = str(img.mode)
            metadata_dict["Dimensions"] = f"{img.width}x{img.height}"

        # Inspect for editing software tags
        software_val = metadata_dict.get("Software", "")
        if isinstance(software_val, str) and software_val.strip():
            for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
                if kw.lower() in software_val.lower():
                    has_editing_software = True
                    editing_software = software_val
                    flags.append(f"Editing software tag identified: '{software_val}'")
                    break

        # Check if basic camera metadata is present
        has_camera_info = "Make" in metadata_dict or "Model" in metadata_dict
        
        # Check if EXIF metadata is stripped (excluding default Pillow Format/Mode/Dimensions)
        exif_keys = [k for k in metadata_dict if k not in ("Format", "Mode", "Dimensions")]
        metadata_stripped = len(exif_keys) == 0

        if metadata_stripped:
            flags.append("EXIF metadata is missing or stripped from the file header.")
        elif not has_camera_info:
            flags.append("Camera Make/Model EXIF fields are missing.")

    except Exception as e:
        flags.append(f"Error inspecting image metadata: {str(e)}")
        metadata_stripped = True

    return MetadataResult(
        metadata_dict=metadata_dict,
        has_editing_software=has_editing_software,
        editing_software=editing_software,
        metadata_stripped=metadata_stripped,
        flags=flags
    )
