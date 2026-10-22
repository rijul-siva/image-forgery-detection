import streamlit as st
import tempfile
import os
import json
import numpy as np
import cv2

from src.preprocessing import preprocess_image
from src.copy_move.pipeline import run_copy_move_analysis
from src.forensics.ela import run_ela_analysis
from src.forensics.metadata import run_metadata_analysis
from src.utils.data_types import ForensicReport

ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def safe_upload_suffix(filename: str) -> str:
    """Return a decoder-safe temporary-file suffix for an uploaded file.

    Preserving the original extension matters: writing a PNG to a `.jpg`
    temp file makes OpenCV/Pillow misread the container, which corrupts
    metadata parsing and the ELA recompression baseline.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in ALLOWED_UPLOAD_EXTENSIONS else ".jpg"


def run_full_pipeline(
    image_path: str, 
    detector: str = "orb",
    ratio: float = 0.75,
    ransac_threshold: float = 3.0,
    ela_quality: int = 95,
    max_edge: int = 1600
) -> ForensicReport:
    """
    Master function that runs the entire PixelGuard pipeline.
    """
    preprocess = preprocess_image(image_path, max_edge=max_edge)
    copy_move = run_copy_move_analysis(
        preprocess.gray_image,
        preprocess.color_image,
        detector=detector,
        ratio=ratio,
        ransac_threshold=ransac_threshold,
        preprocess_scale=preprocess.scale_factor,
    )
    # ELA is scored statistically inside the detected copy-move regions.
    ela = run_ela_analysis(
        image_path,
        quality=ela_quality,
        region_mask=copy_move.mask,
    )
    metadata = run_metadata_analysis(image_path)
    
    return ForensicReport(
        preprocess=preprocess,
        copy_move=copy_move,
        ela=ela,
        metadata=metadata
    )

def report_to_dict(report: ForensicReport) -> dict:
    """Convert ForensicReport to a serializable dictionary for JSON export."""
    cm = report.copy_move
    tp = cm.transform_params
    sf = report.preprocess.scale_factor or 1.0

    tp_dict = None
    if tp is not None:
        tp_dict = {
            "rotation_degrees": round(float(tp.rotation_degrees), 2),
            "scale_factor": round(float(tp.scale_factor), 3),
            "translation_x": round(float(tp.translation_x), 1),
            "translation_y": round(float(tp.translation_y), 1),
            "translation_distance": round(float(tp.translation_distance), 1),
            # Native photo pixels = analysis pixels / scale_factor.
            "translation_x_native": round(float(tp.translation_x) / sf, 1),
            "translation_y_native": round(float(tp.translation_y) / sf, 1),
            "translation_distance_native": round(float(tp.translation_distance) / sf, 1),
        }

    regions_list = []
    for r in cm.regions:
        regions_list.append({
            "bbox": [r.bbox_x, r.bbox_y, r.bbox_width, r.bbox_height],
            "area_px": round(float(r.area_px), 1),
            "area_percent": round(float(r.area_percent), 2),
            "centroid": [round(float(r.centroid_x), 1), round(float(r.centroid_y), 1)],
            "bbox_native": [
                r.bbox_x_native, r.bbox_y_native,
                r.bbox_width_native, r.bbox_height_native,
            ],
            "area_px_native": (
                round(float(r.area_px_native), 1)
                if r.area_px_native is not None else None
            ),
            "centroid_native": [
                round(float(r.centroid_x_native), 1) if r.centroid_x_native is not None else None,
                round(float(r.centroid_y_native), 1) if r.centroid_y_native is not None else None,
            ],
        })
        
    return {
        "preprocessing": {
            "original_width": report.preprocess.original_width,
            "original_height": report.preprocess.original_height,
            "scale_factor": round(float(report.preprocess.scale_factor), 3)
        },
        "copy_move_detection": {
            "forgery_detected": cm.forgery_detected,
            "total_matches": cm.num_total_matches,
            "inlier_matches": cm.num_inlier_matches,
            "confidence_score": round(float(cm.confidence_score), 4),
            "transform_parameters": tp_dict,
            "regions": regions_list
        },
        "ela_analysis": {
            "suspicion_level": report.ela.suspicion_level,
            "description": report.ela.description
        },
        "metadata_analysis": {
            "metadata_stripped": report.metadata.metadata_stripped,
            "has_editing_software": report.metadata.has_editing_software,
            "editing_software": report.metadata.editing_software,
            "flags": report.metadata.flags
        }
    }

def main():
    st.set_page_config(
        page_title="PixelGuard — Image Forgery Detection",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Custom CSS for dark/clean aesthetic
    st.markdown("""
        <style>
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #4F46E5;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #1F2937;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #374151;
            text-align: center;
        }
        .metric-title {
            color: #9CA3AF;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
        }
        .metric-value {
            color: #F3F4F6;
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🛡️ PixelGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Digital Image Forgery Detection, Localization & Parameter Estimation</div>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Pipeline Configuration")
    detector = st.sidebar.selectbox("Feature Detector", ["orb", "sift"], index=0, help="ORB is fast; SIFT is scale & rotation invariant.")
    ratio = st.sidebar.slider("Lowe's Ratio Threshold", 0.50, 0.90, 0.75, 0.05, help="Lower ratio = stricter matching.")
    ransac_threshold = st.sidebar.slider("RANSAC Inlier Threshold (px)", 1.0, 10.0, 3.0, 0.5, help="Maximum reprojection error for inlier matches.")
    ela_quality = st.sidebar.slider("ELA JPEG Quality Level", 80, 100, 95, 1, help="Quality level for JPEG re-compression.")
    max_edge = st.sidebar.select_slider("Max Image Edge (Resolution Cap)", options=[800, 1200, 1600, 2000], value=1600)
    
    uploaded_file = st.file_uploader("Upload Smartphone Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Preserve the uploaded extension so OpenCV / Pillow decode the file
        # with the correct format.
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=safe_upload_suffix(uploaded_file.name)
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            with st.spinner("Analyzing image for forensic indicators..."):
                report = run_full_pipeline(
                    tmp_file_path,
                    detector=detector,
                    ratio=ratio,
                    ransac_threshold=ransac_threshold,
                    ela_quality=ela_quality,
                    max_edge=max_edge
                )
                
            st.success("Analysis Complete!")
            
            # Global Status Indicator
            cm = report.copy_move
            if cm.forgery_detected:
                st.error(f"🚨 **POSSIBLE COPY-MOVE FORGERY DETECTED** (Inlier Match Confidence: {cm.confidence_score:.1%})")
            else:
                st.info("✅ **NO SIGNIFICANT COPY-MOVE FORGERY DETECTED**")
                
            # Quick Metrics Bar
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Status", "DETECTED" if cm.forgery_detected else "CLEAN")
            m2.metric("Inlier Matches", f"{cm.num_inlier_matches} / {cm.num_total_matches}")
            m3.metric("Rotation Angle", f"{cm.transform_params.rotation_degrees:.1f}°" if cm.transform_params else "N/A")
            m4.metric("Scale Factor", f"{cm.transform_params.scale_factor:.2f}×" if cm.transform_params else "N/A")
            m5.metric("Trans. Distance", f"{cm.transform_params.translation_distance:.1f} px" if cm.transform_params else "N/A")
            if cm.transform_params and report.preprocess.scale_factor != 1.0:
                st.caption(
                    f"Native photo coordinates: duplicated content moved "
                    f"{cm.transform_params.translation_distance / report.preprocess.scale_factor:.1f} px "
                    f"(scale factor {report.preprocess.scale_factor:.2f}×)."
                )
            
            st.divider()
            
            # Interactive Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📷 Overview", 
                "🔍 Copy-Move Detection", 
                "⚡ ELA Analysis", 
                "📋 Metadata / EXIF", 
                "💾 Export Report"
            ])
            
            with tab1:
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.subheader("Original Uploaded Image")
                    color_rgb = cv2.cvtColor(report.preprocess.color_image, cv2.COLOR_BGR2RGB)
                    st.image(color_rgb, use_container_width=True)
                    st.caption(f"Original Resolution: {report.preprocess.original_width} × {report.preprocess.original_height} px | Analysis Scale: {report.preprocess.scale_factor:.2f}×")
                    
                with col_b:
                    st.subheader("Executive Summary")
                    st.markdown(f"""
                    - **Forgery Status:** `{"DETECTED" if cm.forgery_detected else "NOT DETECTED"}`
                    - **Confidence Score (Inlier Ratio):** `{cm.confidence_score:.3f}` *(experimental metric)*
                    - **Verified Keypoint Matches:** `{cm.num_inlier_matches}` inliers out of `{cm.num_total_matches}` candidates
                    - **Error Level Analysis (ELA):** Suspicion Level `{report.ela.suspicion_level.upper()}`
                    - **EXIF Metadata Status:** `{"STRIPPED / MISSING" if report.metadata.metadata_stripped else "PRESENT"}`
                    - **Editing Software Tag:** `{report.metadata.editing_software if report.metadata.has_editing_software else "None Detected"}`
                    """)
                    
                    if report.metadata.flags:
                        st.subheader("Forensic Alerts")
                        for flag in report.metadata.flags:
                            st.warning(f"⚠️ {flag}")

            with tab2:
                st.subheader("Copy-Move Forgery Analysis & Quantitative Parameters")
                
                # Visualizations Side-by-Side
                v1, v2 = st.columns(2)
                with v1:
                    if cm.match_visualization is not None:
                        st.image(
                            cv2.cvtColor(cm.match_visualization, cv2.COLOR_BGR2RGB),
                            caption="Keypoint Feature Matches (Green = Inliers, Red = Outliers)",
                            use_container_width=True
                        )
                with v2:
                    if cm.region_visualization is not None:
                        st.image(
                            cv2.cvtColor(cm.region_visualization, cv2.COLOR_BGR2RGB),
                            caption="Localized Suspicious Regions & Direction Vector",
                            use_container_width=True
                        )
                        
                st.divider()
                
                # Geometric Parameters Table
                p1, p2 = st.columns(2)
                with p1:
                    st.subheader("📐 Estimated Transformation Parameters")
                    if cm.transform_params is not None:
                        tp = cm.transform_params
                        sf = report.preprocess.scale_factor or 1.0
                        st.table([
                            {"Parameter": "Rotation Angle (θ)", "Value": f"{tp.rotation_degrees:.2f}°"},
                            {"Parameter": "Scale Factor (s)", "Value": f"{tp.scale_factor:.3f}×"},
                            {"Parameter": "Translation X (dx)", "Value": f"{tp.translation_x:.1f} px"},
                            {"Parameter": "Translation Y (dy)", "Value": f"{tp.translation_y:.1f} px"},
                            {"Parameter": "Euclidean Translation Dist", "Value": f"{tp.translation_distance:.1f} px"},
                            {"Parameter": "Translation Dist (native photo)", "Value": f"{tp.translation_distance / sf:.1f} px"},
                        ])
                        if sf != 1.0:
                            st.caption(
                                f"Analysis canvas is {sf:.2f}× the original; native values are "
                                "divided by the preprocessing scale factor."
                            )
                    else:
                        st.info("No consensus transform found by RANSAC.")
                        
                with p2:
                    st.subheader("📦 Suspicious Region Metrics")
                    if cm.regions:
                        reg_rows = []
                        for idx, reg in enumerate(cm.regions, 1):
                            reg_rows.append({
                                "Region #": idx,
                                "Bounding Box": f"{reg.bbox_width} × {reg.bbox_height} px",
                                "Area (px²)": f"{reg.area_px:.0f}",
                                "Image Area %": f"{reg.area_percent:.2f}%",
                                "Centroid (x,y)": f"({reg.centroid_x:.0f}, {reg.centroid_y:.0f})",
                                "Native BBox": (
                                    f"{reg.bbox_width_native} × {reg.bbox_height_native} px"
                                    if reg.bbox_width_native is not None else "—"
                                ),
                                "Native Centroid": (
                                    f"({reg.centroid_x_native:.0f}, {reg.centroid_y_native:.0f})"
                                    if reg.centroid_x_native is not None else "—"
                                ),
                            })
                        st.table(reg_rows)
                    else:
                        st.info("No localized regions detected.")

            with tab3:
                st.subheader("Error Level Analysis (ELA)")
                st.write("ELA highlights areas with compression level differences. Digitally modified regions often show elevated error levels when re-saved as JPEG.")
                
                e_col1, e_col2 = st.columns([2, 1])
                with e_col1:
                    ela_rgb = cv2.cvtColor(report.ela.ela_image, cv2.COLOR_BGR2RGB)
                    st.image(ela_rgb, caption=f"ELA Heatmap (Quality: {ela_quality}%)", use_container_width=True)
                with e_col2:
                    st.markdown("### ELA Summary")
                    st.markdown(f"**Suspicion Level:** `{report.ela.suspicion_level.upper()}`")
                    st.info(report.ela.description)
                    st.caption("Suspicion is scored statistically: when copy-move regions were found, the error inside those regions is compared with the background (in background-σ units); otherwise a localized block z-score is used. Sharp natural edges are no longer flagged by a raw global maximum.")

            with tab4:
                st.subheader("EXIF Metadata Inspection")
                m_col1, m_col2 = st.columns([1, 1])
                with m_col1:
                    st.markdown("### Metadata Summary")
                    st.write(f"**Metadata Stripped:** `{report.metadata.metadata_stripped}`")
                    st.write(f"**Editing Software Flag:** `{report.metadata.has_editing_software}`")
                    if report.metadata.has_editing_software:
                        st.write(f"**Software Detected:** `{report.metadata.editing_software}`")
                        
                    if report.metadata.flags:
                        st.markdown("### Detected Flags")
                        for flag in report.metadata.flags:
                            st.warning(flag)
                            
                with m_col2:
                    st.markdown("### Raw EXIF Tags")
                    if report.metadata.metadata_dict:
                        with st.expander("View All Extracted Tags", expanded=True):
                            st.json(report.metadata.metadata_dict)
                    else:
                        st.info("No EXIF metadata tags found in file header.")

            with tab5:
                st.subheader("Export Forensic Findings")
                st.write("Download the consolidated forensic report as a structured JSON document.")
                report_json = json.dumps(report_to_dict(report), indent=2)
                st.download_button(
                    label="📥 Download Forensic Report (JSON)",
                    data=report_json,
                    file_name="pixelguard_forensic_report.json",
                    mime="application/json"
                )
                st.json(report_to_dict(report))

        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

if __name__ == "__main__":
    main()
