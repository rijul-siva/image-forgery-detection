import streamlit as st
import tempfile
import os
import numpy as np
import cv2

from src.preprocessing import preprocess_image
from src.forensics.ela import run_ela_analysis
from src.forensics.metadata import run_metadata_analysis
from src.utils.data_types import (
    ForensicReport, 
    CopyMoveResult, 
    TransformParams, 
    RegionInfo, 
    PreprocessResult
)

# Dummy copy-move analysis function for integration testing
def mock_run_copy_move_analysis(gray_image: np.ndarray, color_image: np.ndarray) -> CopyMoveResult:
    return CopyMoveResult(
        forgery_detected=False,
        num_total_matches=0,
        num_inlier_matches=0,
        confidence_score=0.0,
        inlier_keypoints_src=[],
        inlier_keypoints_dst=[],
        affine_matrix=None,
        transform_params=None,
        regions=[],
        mask=None,
        match_visualization=None,
        region_visualization=None
    )

def run_full_pipeline(image_path: str, detector: str = "orb") -> ForensicReport:
    """
    Master function that runs the entire PixelGuard pipeline.
    """
    preprocess = preprocess_image(image_path)
    copy_move = mock_run_copy_move_analysis(preprocess.gray_image, preprocess.color_image)
    ela = run_ela_analysis(image_path)
    metadata = run_metadata_analysis(image_path)
    
    return ForensicReport(
        preprocess=preprocess,
        copy_move=copy_move,
        ela=ela,
        metadata=metadata
    )

def main():
    st.set_page_config(page_title="PixelGuard", layout="wide")
    st.title("PixelGuard: Image Forgery Detection")
    st.write("Upload an image to run forensics analysis.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            with st.spinner("Running forensics pipeline..."):
                report = run_full_pipeline(tmp_file_path)
                
            st.success("Analysis Complete!")
            
            # Create tabs for different results
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Copy-Move", "ELA", "Metadata"])
            
            with tab1:
                st.subheader("Original Image")
                # Convert BGR to RGB for Streamlit
                color_image_rgb = cv2.cvtColor(report.preprocess.color_image, cv2.COLOR_BGR2RGB)
                st.image(color_image_rgb, use_column_width=True)
                st.write(f"Original Dimensions: {report.preprocess.original_width}x{report.preprocess.original_height}")
                
            with tab2:
                st.subheader("Copy-Move Detection")
                st.write(f"Forgery Detected: {report.copy_move.forgery_detected}")
                # Mock data doesn't have much to show, but we can add placeholders
                
            with tab3:
                st.subheader("Error Level Analysis")
                ela_image_rgb = cv2.cvtColor(report.ela.ela_image, cv2.COLOR_BGR2RGB)
                st.image(ela_image_rgb, use_column_width=True)
                st.write(f"Suspicion Level: {report.ela.suspicion_level}")
                st.write(report.ela.description)
                
            with tab4:
                st.subheader("Metadata Analysis")
                st.write(f"Metadata Stripped: {report.metadata.metadata_stripped}")
                st.write(f"Editing Software Found: {report.metadata.has_editing_software}")
                if report.metadata.has_editing_software:
                    st.write(f"Software: {report.metadata.editing_software}")
                if report.metadata.flags:
                    st.write("Flags:")
                    for flag in report.metadata.flags:
                        st.warning(flag)
                
                with st.expander("View Raw Metadata"):
                    st.json(report.metadata.metadata_dict)
                    
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

if __name__ == "__main__":
    main()
