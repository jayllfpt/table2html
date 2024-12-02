import cv2
import streamlit as st
from table2html import Table2HTML
from table2html.source import visualize_boxes, crop_image
import numpy as np
import time
import os
import tempfile
import fitz  # PyMuPDF
from PIL import Image


def initialize_session_state():
    if 'table_detections' not in st.session_state:
        st.session_state.table_detections = []
    if 'structure_detections' not in st.session_state:
        st.session_state.structure_detections = []
    if 'cropped_tables' not in st.session_state:
        st.session_state.cropped_tables = []
    if 'html_tables' not in st.session_state:
        st.session_state.html_tables = []
    if 'detection_data' not in st.session_state:
        st.session_state.detection_data = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None


def clear_results():
    st.session_state.table_detections = []
    st.session_state.structure_detections = []
    st.session_state.cropped_tables = []
    st.session_state.html_tables = []

def process_image(image, model_path, table_crop_padding):
    table2html = Table2HTML(table_detection_model_path=model_path)
    detection_data = table2html(image, table_crop_padding)

    if len(detection_data) == 0:
        st.warning("No tables detected on this page.")
        return

    # Clear previous results
    st.session_state.detection_data = detection_data

    for data in detection_data:
        # Store table detection visualization
        table_detection = visualize_boxes(
            image.copy(),
            [data["table_bbox"]],
            color=(0, 0, 255),
            thickness=2
        )
        st.session_state.table_detections.append(table_detection)

        # Store cropped table
        cropped_table = crop_image(
            image, data["table_bbox"], table_crop_padding)
        st.session_state.cropped_tables.append(cropped_table)

        # Store structure detection visualization
        structure_detection = visualize_boxes(
            cropped_table.copy(),
            [cell['box'] for cell in data['cells']],
            color=(0, 255, 0),
            thickness=1
        )
        st.session_state.structure_detections.append(structure_detection)

        # Store HTML
        st.session_state.html_tables.append(data["html"])


def inference_one_image(image, model_paths, table_crop_padding):
    clear_results()
    with st.spinner("Processing..."):
        start_time = time.time()

        try:
            # Update process_image call to include all model paths
            process_image(
                image, model_paths["table_detection"], table_crop_padding)

            # Clean up temporary files if using custom models
            for model_type, path in model_paths.items():
                if f"{model_type}_option" in st.session_state and \
                        st.session_state[f"{model_type}_option"] == "custom":
                    os.unlink(path)

            execution_time = time.time() - start_time
            st.success(
                f"Processing completed in {execution_time:.2f} seconds")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            # Clean up temporary files on error
            for model_type, path in model_paths.items():
                if f"{model_type}_option" in st.session_state and \
                        st.session_state[f"{model_type}_option"] == "custom":
                    os.unlink(path)


def main():
    initialize_session_state()
    st.set_page_config(layout="wide")

    # Create columns for the top layout
    col1, col2 = st.columns([1, 1])

    # Left container - Image Upload
    with col1:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image or PDF file",
            type=['jpg', 'jpeg', 'png', 'pdf']
        )


    # Right container - Configuration
    with col2:
        st.subheader("Configuration")

        # Create columns for model configurations
        model_cols = st.columns(3)  # Create 3 equal columns
        model_paths = {}

        # Model selection options
        model_types = ["Table Detection", "Column Detection", "Row Detection"]

        for idx, model_type in enumerate(model_types):
            with model_cols[idx]:  # Use each column for a model type
                key_prefix = model_type.lower().replace(" ", "_")
                st.markdown(f"**{model_type} Model**")

                model_option = st.radio(
                    f"Choose {model_type} Model",
                    options=["default", "custom"],
                    horizontal=True,
                    key=f"{key_prefix}_option"
                )

                if model_option == "default":
                    default_path = f"models/{key_prefix}.pt"
                    model_paths[key_prefix] = default_path
                    st.info(f"Using default model: {default_path}")
                else:
                    model_upload = st.file_uploader(
                        f"Choose {model_type} Model File (.pt)",
                        type=['pt'],
                        key=f"{key_prefix}_upload"
                    )
                    if model_upload:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                            tmp_file.write(model_upload.getvalue())
                            model_paths[key_prefix] = tmp_file.name
                    else:
                        model_paths[key_prefix] = None
                        st.warning(
                            f"Please upload a {model_type.lower()} model file")

        # Padding input below the model configurations
        st.markdown("---")  # Add a separator
        table_crop_padding = st.number_input(
            "Table Crop Padding",
            value=15,
            min_value=0,
            max_value=100
        )

    current_image = None

    if uploaded_file is not None and all(model_paths.values()):
        if uploaded_file.type == "application/pdf":
            # Convert PDF to images
            pdf_bytes = uploaded_file.read()
            pdf_images = []

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=200)
                pil_image = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(pil_image)
                pdf_images.append(img_array)

            # Show thumbnails
            st.write("Select a page to process:")
            cols = st.columns(4)
            for idx, img in enumerate(pdf_images):
                with cols[idx % 4]:
                    st.image(img, caption=f"Page {idx+1}", width=150)
                    if st.button(f"Process Page {idx+1}"):
                        current_image = img
                        st.session_state.current_image = img
                        inference_one_image(
                            current_image, model_paths, table_crop_padding)
        else:
            # Handle regular image upload
            file_bytes = np.asarray(
                bytearray(uploaded_file.read()), dtype=np.uint8)
            current_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.current_image = current_image

            # Process button
            if st.button("Process Image"):
                inference_one_image(current_image, model_paths, table_crop_padding)


    if len(st.session_state.cropped_tables) > 0:
        st.header("Results")

        # General Results Section
        st.subheader("General Results")
        gen_img_col, gen_html_col = st.columns([1, 1])

        with gen_img_col:
            show_all_detections = st.toggle(
                "Show Table Detections",
                value=False,
                key="show_all_detections"
            )

            # Display either original image or detection visualization
            if show_all_detections and len(st.session_state.detection_data) > 0:
                # Create visualization with all table detections
                all_tables_viz = visualize_boxes(
                    st.session_state.current_image.copy(),
                    [data["table_bbox"]
                        for data in st.session_state.detection_data],
                    color=(0, 0, 255),
                    thickness=2
                )
                st.image(
                    all_tables_viz,
                    caption="All Table Detections",
                    use_container_width=True
                )
            else:
                st.image(
                    st.session_state.current_image,
                    caption="Original Image",
                    use_container_width=True
                )

        with gen_html_col:
            st.markdown("### All HTML Tables:")
            # Combine all HTML tables
            all_html = "\n".join(st.session_state.html_tables)
            st.markdown(all_html, unsafe_allow_html=True)

            # Download all HTML tables
            combined_html = "<!DOCTYPE html><html><body>\n" + all_html + "\n</body></html>"
            st.download_button(
                label="Download All Tables HTML",
                data=combined_html,
                file_name="all_tables.html",
                mime="text/html",
                key="download_all_btn"
            )

        st.divider()

        # Detailed Results Section
        show_details = st.toggle("Show Detailed Results", value=False)

        if show_details:
            st.subheader("Detailed Results")
            for idx in range(len(st.session_state.cropped_tables)):
                st.subheader(f"Table {idx + 1}")

                # Visualization controls for each table
                control_col1, control_col2 = st.columns([1, 1])
                with control_col1:
                    show_table_detection = st.toggle(
                        f"Show Table Detection for Table {idx + 1}",
                        value=False,
                        key=f"table_detection_{idx}"
                    )
                with control_col2:
                    show_structure_detection = st.toggle(
                        f"Show Structure Detection for Table {idx + 1}",
                        value=False,
                        key=f"structure_detection_{idx}"
                    )

                # Create columns for each table result
                img_col, html_col = st.columns([1, 1])

                with img_col:
                    # Show either the cropped table or visualizations based on toggles
                    if show_table_detection:
                        st.image(
                            st.session_state.table_detections[idx],
                            caption="Table Detection",
                            use_container_width=True
                        )
                    if show_structure_detection:
                        st.image(
                            st.session_state.structure_detections[idx],
                            caption="Structure Detection",
                            use_container_width=True
                        )
                    if not show_table_detection and not show_structure_detection:
                        st.image(
                            st.session_state.cropped_tables[idx],
                            caption="Cropped Table",
                            use_container_width=True
                        )

                with html_col:
                    st.markdown("### HTML Output:")
                    st.markdown(
                        st.session_state.html_tables[idx],
                        unsafe_allow_html=True
                    )
                    st.download_button(
                        label=f"Download Table {idx + 1} HTML",
                        data=st.session_state.html_tables[idx],
                        file_name=f"table_{idx + 1}.html",
                        mime="text/html",
                        key=f"download_btn_{idx}"
                    )

                st.divider()


if __name__ == "__main__":
    main()
