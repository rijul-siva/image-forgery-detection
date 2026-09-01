import streamlit as st
import tempfile
import os
import cv2

from src.preprocessing import preprocess_image
from src.forensics.ela import run_ela_analysis
from src.forensics.metadata import run_metadata_analysis
from src.copy_move.pipeline import run_copy_move_analysis
from src.utils.data_types import ForensicReport


def run_full_pipeline(image_path: str, detector: str = "orb") -> ForensicReport:
    """
    Master function that runs the entire PixelGuard pipeline.
    """
    preprocess = preprocess_image(image_path)
    copy_move = run_copy_move_analysis(
        preprocess.gray_image,
        preprocess.color_image,
        detector=detector,
    )
    ela = run_ela_analysis(image_path)
    metadata = run_metadata_analysis(image_path)

    return ForensicReport(
        preprocess=preprocess,
        copy_move=copy_move,
        ela=ela,
        metadata=metadata,
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

            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Copy-Move", "ELA", "Metadata"])

            with tab1:
                st.subheader("Original Image")
                color_image_rgb = cv2.cvtColor(report.preprocess.color_image, cv2.COLOR_BGR2RGB)
                st.image(color_image_rgb, use_container_width=True)
                st.write(
                    f"Original Dimensions: {report.preprocess.original_width}"
                    f"x{report.preprocess.original_height}"
                )

            with tab2:
                st.subheader("Copy-Move Detection")
                cm = report.copy_move
                st.write(f"**Forgery Detected:** `{cm.forgery_detected}`")
                st.write(
                    f"**Matches:** {cm.num_inlier_matches} inliers / "
                    f"{cm.num_total_matches} total "
                    f"(confidence: {cm.confidence_score:.3f} -- experimental)"
                )

                if cm.transform_params is not None:
                    tp = cm.transform_params
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rotation", f"{tp.rotation_degrees:.2f} deg")
                    c2.metric("Scale", f"{tp.scale_factor:.3f}x")
                    c3.metric("Translation", f"{tp.translation_distance:.1f} px")
                    st.write(
                        f"Translation vector: ({tp.translation_x:.1f}, "
                        f"{tp.translation_y:.1f}) px"
                    )
                else:
                    st.info("No geometric transform estimated (no consensus).")

                if cm.regions:
                    st.write(f"**Suspicious regions:** {len(cm.regions)}")
                    rows = []
                    for i, r in enumerate(cm.regions):
                        rows.append(
                            {
                                "#": i + 1,
                                "bbox": f"{r.bbox_width}x{r.bbox_height}",
                                "area %": f"{r.area_percent:.2f}",
                                "centroid": f"({r.centroid_x:.0f}, {r.centroid_y:.0f})",
                            }
                        )
                    st.table(rows)
                else:
                    st.info("No suspicious regions localized.")

                c4, c5 = st.columns(2)
                with c4:
                    if cm.match_visualization is not None:
                        st.image(
                            cv2.cvtColor(cm.match_visualization, cv2.COLOR_BGR2RGB),
                            caption="Feature matches (green = inlier, red = outlier)",
                            use_container_width=True,
                        )
                with c5:
                    if cm.region_visualization is not None:
                        st.image(
                            cv2.cvtColor(cm.region_visualization, cv2.COLOR_BGR2RGB),
                            caption="Localized suspicious regions",
                            use_container_width=True,
                        )

            with tab3:
                st.subheader("Error Level Analysis")
                ela_image_rgb = cv2.cvtColor(report.ela.ela_image, cv2.COLOR_BGR2RGB)
                st.image(ela_image_rgb, use_container_width=True)
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
