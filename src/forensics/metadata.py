import os
from PIL import Image
from PIL.ExifTags import TAGS
from src.utils.data_types import MetadataResult

def run_metadata_analysis(image_path: str) -> MetadataResult:
    """
    Extract and analyze EXIF/image metadata.
    """
    metadata_dict = {}
    has_editing_software = False
    editing_software = None
    flags = []
    
    suspicious_software_keywords = ["photoshop", "gimp", "lightroom", "canva", "snapseed"]
    
    try:
        image = Image.open(image_path)
        exifdata = image.getexif()
        
        if exifdata:
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    try:
                        data = data.decode()
                    except:
                        pass
                metadata_dict[tag] = data
                
            # Check software
            software = metadata_dict.get('Software', '')
            if isinstance(software, str) and software.strip():
                for keyword in suspicious_software_keywords:
                    if keyword.lower() in software.lower():
                        has_editing_software = True
                        editing_software = software
                        flags.append(f"Suspicious editing software found: {software}")
                        break
        
        metadata_stripped = len(metadata_dict) == 0
        if metadata_stripped:
            flags.append("Metadata is stripped/missing.")
            
    except Exception as e:
        flags.append(f"Error reading metadata: {str(e)}")
        metadata_stripped = True

    return MetadataResult(
        metadata_dict=metadata_dict,
        has_editing_software=has_editing_software,
        editing_software=editing_software,
        metadata_stripped=metadata_stripped,
        flags=flags
    )
