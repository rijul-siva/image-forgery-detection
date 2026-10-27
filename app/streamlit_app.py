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


def fit_dip_preview(image: np.ndarray, max_width: int = 900, max_height: int = 520) -> np.ndarray:
    """Scale a DIP preview to fit a screen while preserving its aspect ratio."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


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
    max_edge: int = 1600,
    enhance_clahe: bool = False,
    clahe_clip_limit: float = 2.0
) -> ForensicReport:
    """
    Master function that runs the entire PixelGuard pipeline.
    """
    preprocess = preprocess_image(
        image_path, 
        max_edge=max_edge, 
        enhance_clahe=enhance_clahe, 
        clahe_clip_limit=clahe_clip_limit
    )
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
            "regions": regions_list,
            "displacement_clusters": cm.num_displacement_clusters,
            "ignored_periodic_lattices": list(cm.periodic_lattices),
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
    
    st.sidebar.subheader("🔬 DIP Enhancements")
    enhance_clahe = st.sidebar.checkbox(
        "Enhance Contrast (CLAHE)", 
        value=False, 
        help="Contrast Limited Adaptive Histogram Equalization boosts local contrast in shadowy or low-contrast regions."
    )
    clahe_clip_limit = (
        st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.5, help="Higher = more contrast amplification.")
        if enhance_clahe else 2.0
    )
    
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
                    max_edge=max_edge,
                    enhance_clahe=enhance_clahe,
                    clahe_clip_limit=clahe_clip_limit
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
            tab1, tab_dip, tab3, tab4, tab5 = st.tabs([
                "📷 Overview", 
                "🔬 Step-by-Step DIP Pipeline",
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

            with tab_dip:
                st.subheader("🔬 Digital Image Processing (DIP) — Step-by-Step Execution")
                st.write(
                    "This section displays the intermediate image output and mathematical formulation "
                    "at each discrete Digital Image Processing stage of the forensic pipeline."
                )
                
                # --- DIP STAGE 1: Preprocessing & Spatial Filtering ---
                with st.expander("📍 DIP Stage 1: Spatial Sampling, Grayscale Conversion & Denoising", expanded=True):
                    st.markdown("""
                    **DIP Concepts:** Color-to-grayscale intensity conversion via luminance weighting 
                    $Y = 0.299R + 0.587G + 0.114B$, decimation via area interpolation (`cv2.INTER_AREA`), 
                    high-frequency noise suppression via 2D Gaussian convolution $G(x,y,\\sigma)$, and 
                    **local adaptive contrast enhancement via CLAHE** (Contrast Limited Adaptive Histogram Equalization).
                    """)
                    d1_col1, d1_col2, d1_col3, d1_col4 = st.columns(4)
                    with d1_col1:
                        st.image(
                            cv2.cvtColor(report.preprocess.color_image, cv2.COLOR_BGR2RGB),
                            caption=f"1.1 Resized Color ({report.preprocess.color_image.shape[1]}×{report.preprocess.color_image.shape[0]} px)",
                            use_container_width=True
                        )
                    with d1_col2:
                        if report.preprocess.denoised_image is not None:
                            st.image(
                                cv2.cvtColor(report.preprocess.denoised_image, cv2.COLOR_BGR2RGB),
                                caption="1.2 Gaussian Filtered (3×3, σ=0.5)",
                                use_container_width=True
                            )
                        else:
                            st.info("Gaussian smoothing disabled in current run.")
                    with d1_col3:
                        st.image(
                            report.preprocess.gray_image,
                            caption="1.3 Grayscale Luminance (1-Channel uint8)",
                            use_container_width=True
                        )
                    with d1_col4:
                        if report.preprocess.clahe_image is not None:
                            st.image(
                                report.preprocess.clahe_image,
                                caption="1.4 CLAHE Enhanced (Contextual Hist. Eq.)",
                                use_container_width=True
                            )
                        else:
                            st.info("CLAHE enhancement disabled.")

                    # Grayscale Histogram
                    hist, _ = np.histogram(report.preprocess.gray_image, bins=32, range=(0, 256))
                    st.caption("Grayscale Intensity Distribution Histogram (32 bins):")
                    st.bar_chart(hist)

                # --- DIP STAGE 2: Feature Extraction ---
                with st.expander("📍 DIP Stage 2: Scale-Space Keypoint & Interest Point Detection", expanded=True):
                    st.markdown("""
                    **DIP Concepts:** Scale-space multi-scale representation (SIFT Difference-of-Gaussians pyramid / ORB FAST corners with Harris scoring). 
                    Each keypoint is localized with coordinate $(x,y)$, characteristic scale circle radius $\\sigma$, and dominant gradient orientation $\\theta$.
                    """)
                    d2_col1, d2_col2 = st.columns([2, 1])
                    with d2_col1:
                        if cm.keypoints_visualization is not None:
                            st.image(
                                fit_dip_preview(cv2.cvtColor(cm.keypoints_visualization, cv2.COLOR_BGR2RGB)),
                                caption=f"2.1 Detected Scale-Space Keypoints ({detector.upper()}) with Scale Circles & Orientation Vectors",
                                width="content"
                            )
                    with d2_col2:
                        st.markdown("### Keypoint Metrics")
                        st.markdown(f"- **Detector Architecture:** `{detector.upper()}`")
                        st.markdown(f"- **Total Keypoints Detected:** `{len(cm.inlier_keypoints_src) + len(cm.inlier_keypoints_dst)} verified points`")
                        st.markdown(f"- **Descriptor Dimensions:** `{'128 continuous floats (SIFT)' if detector == 'sift' else '32 binary bytes / 256 bits (ORB)'}`")
                        st.markdown(f"- **Distance Metric:** `{'L2 Euclidean Norm' if detector == 'sift' else 'Hamming Distance'}`")

                # --- DIP STAGE 3: Feature Matching & Filtering ---
                with st.expander("📍 DIP Stage 3: Feature Descriptor Matching & Spatial Suppression", expanded=True):
                    st.markdown("""
                    **DIP Concepts:** $k$-Nearest-Neighbor matching ($k=3$), Lowe's ratio test $\\frac{d_1}{d_2} < \\tau$ to eliminate ambiguous matches, 
                    and spatial distance thresholding $\\|p_i - p_j\\|_2 \\ge d_{\\text{min}}$ to suppress trivial self-matches.
                    """)
                    d3_col1, d3_col2 = st.columns([2, 1])
                    with d3_col1:
                        if cm.raw_matches_visualization is not None:
                            st.image(
                                fit_dip_preview(cv2.cvtColor(cm.raw_matches_visualization, cv2.COLOR_BGR2RGB)),
                                caption="3.1 Candidate Feature Matches Surviving Ratio & Spatial Suppression",
                                width="content"
                            )
                    with d3_col2:
                        st.markdown("### Matching Parameters")
                        st.markdown(f"- **Candidate Match Pairs:** `{cm.num_total_matches}`")
                        st.markdown(f"- **Lowe's Ratio Threshold:** `{ratio:.2f}`")
                        st.markdown("- **Spatial Suppression Radius:** `10.0 px`")

                # --- DIP STAGE 4: Geometric Verification & RANSAC ---
                with st.expander("📍 DIP Stage 4: Geometric Transformation Estimation & RANSAC Consensus", expanded=True):
                    st.markdown("""
                    **DIP Concepts:** RANSAC (Random Sample Consensus) affine model fitting to separate consensus transformation inliers from spurious background matches:
                    $$\\begin{bmatrix} x' \\\\ y' \\end{bmatrix} = \\begin{bmatrix} a_{00} & a_{01} \\\\ a_{10} & a_{11} \\end{bmatrix} \\begin{bmatrix} x \\\\ y \\end{bmatrix} + \\begin{bmatrix} t_x \\\\ t_y \\end{bmatrix}$$
                    Decomposed into physical rotation $\\theta = \\operatorname{atan2}(a_{10}, a_{00})$, scale factor $s = \\sqrt{\\det(A)}$, and translation distance $d = \\sqrt{t_x^2 + t_y^2}$.
                    """)
                    d4_col1, d4_col2 = st.columns([2, 1])
                    with d4_col1:
                        if cm.match_visualization is not None:
                            st.image(
                                fit_dip_preview(cv2.cvtColor(cm.match_visualization, cv2.COLOR_BGR2RGB)),
                                caption="4.1 RANSAC Classification (Green = Geometric Inliers, Red = Outliers)",
                                width="content"
                            )
                    with d4_col2:
                        st.markdown("### RANSAC Decomposed Parameters")
                        if cm.transform_params is not None:
                            st.markdown(f"- **Inlier / Total Matches:** `{cm.num_inlier_matches} / {cm.num_total_matches}` ({cm.confidence_score:.1%})")
                            st.markdown(f"- **Rotation (θ):** `{cm.transform_params.rotation_degrees:.2f}°`")
                            st.markdown(f"- **Scale Factor (s):** `{cm.transform_params.scale_factor:.3f}×`")
                            st.markdown(f"- **Translation (dx, dy):** `({cm.transform_params.translation_x:.1f}, {cm.transform_params.translation_y:.1f}) px`")
                            st.markdown(f"- **Euclidean Distance:** `{cm.transform_params.translation_distance:.1f} px`")
                        else:
                            st.info("No consensus geometric transformation found.")

                        # Repetitive UI structures (a table whose columns
                        # repeat every N pixels) are recognised as lattices
                        # and excluded from the forgery verdict.
                        if cm.periodic_lattices:
                            st.markdown("### Ignored Periodic Lattices")
                            for lat in cm.periodic_lattices:
                                harmonics = ", ".join(str(h) for h in lat.get("harmonics", []))
                                st.markdown(
                                    f"- **Base step:** `{lat.get('base_step_x', 0):.1f}, {lat.get('base_step_y', 0):.1f} px` "
                                    f"(length `{lat.get('base_step_length', 0):.1f} px`) — "
                                    f"harmonics `{harmonics}` — `{lat.get('num_matches', 0)}` matches"
                                )
                            st.caption(
                                "These shift vectors repeat at a constant step, which is the "
                                "signature of a repetitive interface (e.g. a weather table) "
                                "rather than of a duplicated region."
                            )
                        if cm.num_displacement_clusters:
                            st.caption(
                                f"Matched against {cm.num_displacement_clusters} distinct "
                                "shift-vector groups."
                            )

                # --- DIP STAGE 5: Mathematical Morphology & Segmentation ---
                with st.expander("📍 DIP Stage 5: Binary Morphology & Spatial Region Segmentation", expanded=True):
                    st.markdown("""
                    **DIP Concepts:** Mathematical morphology on binary keypoint masks:
                    1. **Raw Mask:** Keypoint footprint discs drawn at verified inlier positions.
                    2. **Morphological Closing:** $A \\bullet B = (A \\oplus B) \\ominus B$ (dilation followed by erosion with $7\\times 7$ ellipse) to fuse interior gaps.
                    3. **Morphological Dilation:** $A \\oplus B$ to expand coverage to object boundaries.
                    4. **Connected Components & Contours:** Green's theorem area integration $\\iint dx dy$ and centroid spatial moments $m_{10}/m_{00}, m_{01}/m_{00}$.
                    """)
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    with m_col1:
                        if cm.raw_mask is not None:
                            st.image(cm.raw_mask, caption="5.1 Raw Keypoint Binary Mask", use_container_width=True)
                        else:
                            st.caption("No mask generated.")
                    with m_col2:
                        if cm.closed_mask is not None:
                            st.image(cm.closed_mask, caption="5.2 Morphological Closing (A • B)", use_container_width=True)
                        else:
                            st.caption("No closed mask.")
                    with m_col3:
                        if cm.dilated_mask is not None:
                            st.image(cm.dilated_mask, caption="5.3 Morphological Dilation (A ⊕ B)", use_container_width=True)
                        else:
                            st.caption("No dilated mask.")
                    with m_col4:
                        if cm.region_visualization is not None:
                            st.image(
                                cv2.cvtColor(cm.region_visualization, cv2.COLOR_BGR2RGB),
                                caption="5.4 Contours & Displacement Vector",
                                use_container_width=True
                            )

                # --- DIP STAGE 6: Error Level Analysis (JPEG Compression Forensics) ---
                with st.expander("📍 DIP Stage 6: Error Level Analysis (JPEG Compression Forensics)", expanded=True):
                    st.markdown("""
                    **DIP Concepts:** Frequency-domain Discrete Cosine Transform (DCT) lossy compression analysis. 
                    Re-saving at quality level $Q=95$ induces error discrepancy $\\Delta(x,y) = |I(x,y) - I_{\\text{JPEG}}(x,y)|$. 
                    Digital forgeries present localized quantization variance against the uniform background error floor.
                    """)
                    e1, e2, e3, e4 = st.columns(4)
                    with e1:
                        if report.ela.compressed_image is not None:
                            st.image(
                                cv2.cvtColor(report.ela.compressed_image, cv2.COLOR_BGR2RGB),
                                caption=f"6.1 Re-compressed (JPEG Q={ela_quality})",
                                use_container_width=True
                            )
                    with e2:
                        if report.ela.raw_diff is not None:
                            st.image(
                                cv2.cvtColor(report.ela.raw_diff, cv2.COLOR_BGR2RGB),
                                caption="6.2 Absolute Difference |I - I_JPEG|",
                                use_container_width=True
                            )
                    with e3:
                        st.image(
                            cv2.cvtColor(report.ela.ela_image, cv2.COLOR_BGR2RGB),
                            caption="6.3 Amplified Error (×20)",
                            use_container_width=True
                        )
                    with e4:
                        if report.ela.colormap_image is not None:
                            st.image(
                                cv2.cvtColor(report.ela.colormap_image, cv2.COLOR_BGR2RGB),
                                caption="6.4 Jet Colormap Pseudocolor",
                                use_container_width=True
                            )

                st.info("💡 **Tip for Review / Viva:** You can also run `python scripts/export_dip_stages.py` in your terminal to save all 14 individual DIP stage images to the `outputs/dip_stages/` folder.")
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
