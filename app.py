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


default_configs = {
    'table_detection': {
        'model_path': 'models/table_detection.pt',
        'confidence_threshold': 0.25,
        'iou_threshold': 0.7
    },
    'column_detection': {
        'model_path': 'models/column_detection.pt',
        'confidence_threshold': 0.25,
        'iou_threshold': 0.7,
        'task': 'detect'
    },
    'row_detection': {
        'model_path': 'models/row_detection.pt',
        'confidence_threshold': 0.25,
        'iou_threshold': 0.7,
        'task': 'detect'
    },
    'table_crop_padding': 15
}

thumbnail_columns = 5


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
    if 'configs' not in st.session_state:
        st.session_state.configs = default_configs


def clear_results():
    st.session_state.table_detections = []
    st.session_state.structure_detections = []
    st.session_state.cropped_tables = []
    st.session_state.html_tables = []


def detect_update_results(image, configs):
    table2html = Table2HTML(
        table_detection_config=configs["table_detection"],
        row_detection_config=configs["row_detection"],
        column_detection_config=configs["column_detection"]
    )
    detection_data = table2html(image, configs["table_crop_padding"])

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
            image, data["table_bbox"], configs["table_crop_padding"])
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


def inference_one_image(image, configs):
    clear_results()
    with st.spinner("Processing..."):
        start_time = time.time()

        try:
            # Update process_image call to include all model paths
            detect_update_results(image, configs)

            # Clean up temporary files if using custom models
            for model_type, config in configs.items():
                if f"{model_type}_option" in st.session_state and \
                        st.session_state[f"{model_type}_option"] == "custom":
                    os.unlink(config["model_path"])

            execution_time = time.time() - start_time
            st.success(
                f"Processing completed in {execution_time:.2f} seconds")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            # Clean up temporary files on error
            for model_type, config in configs.items():
                if f"{model_type}_option" in st.session_state and \
                        st.session_state[f"{model_type}_option"] == "custom":
                    os.unlink(config["model_path"])


def main():
    initialize_session_state()
    st.set_page_config(layout="wide")

    # Add page selection
    page = st.sidebar.radio("Select Page", ["Inference", "Configuration"])

    if page == "Inference":
        st.title("Table Detection and Recognition")

        # Image Upload Section
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image or PDF file",
            type=['jpg', 'jpeg', 'png', 'pdf']
        )

        # Get configurations from session state
        configs = st.session_state.get('configs', default_configs)

        current_image = None

        if uploaded_file is not None and all(configs.values()):
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
                cols = st.columns(thumbnail_columns)
                for idx, img in enumerate(pdf_images):
                    with cols[idx % thumbnail_columns]:
                        st.image(img, width=150, use_container_width=True)
                        if st.button(f"Process Page {idx+1}"):
                            current_image = img
                            st.session_state.current_image = img
                            inference_one_image(
                                current_image, configs)
            else:
                # Handle regular image upload
                file_bytes = np.asarray(
                    bytearray(uploaded_file.read()), dtype=np.uint8)
                current_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.current_image = current_image

                # Process button
                if st.button("Process Image"):
                    inference_one_image(
                        current_image, configs)

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

    else:  # Configuration page
        st.title("Model Configuration")

        # Model selection options
        model_types = ["Table Detection", "Column Detection", "Row Detection"]
        configs = {}  # Store both paths and thresholds

        for idx, model_type in enumerate(model_types):
            st.markdown(f"### {model_type}")
            key_prefix = model_type.lower().replace(" ", "_")

            # Model file selection
            model_option = st.radio(
                f"Choose {model_type} Model",
                options=["default", "custom"],
                horizontal=True,
                key=f"{key_prefix}_option"
            )

            if model_option == "default":
                default_path = f"models/{key_prefix}.pt"
                configs[key_prefix] = {"model_path": default_path}
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
                        configs[key_prefix] = {
                            "model_path": tmp_file.name}
                else:
                    configs[key_prefix] = {"model_path": None}
                    st.warning(
                        f"Please upload a {model_type.lower()} model file")

            # Add threshold controls
            thresh_col1, thresh_col2 = st.columns(2)
            with thresh_col1:
                conf_threshold = st.slider(
                    f"{model_type} Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    step=0.05,
                    key=f"{key_prefix}_conf_threshold"
                )
            with thresh_col2:
                iou_threshold = st.slider(
                    f"{model_type} IOU Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key=f"{key_prefix}_iou_threshold"
                )

            if configs[key_prefix]["model_path"]:
                configs[key_prefix].update({
                    "confidence_threshold": conf_threshold,
                    "iou_threshold": iou_threshold
                })
                # Add task field for row and column detection
                if key_prefix in ["column_detection", "row_detection"]:
                    configs[key_prefix]["task"] = "detect"

            st.divider()

        # Padding input below the model configurations
        table_crop_padding = st.number_input(
            "Table Crop Padding",
            value=15,
            min_value=0,
            max_value=100
        )

        # Save configurations to session state
        if st.button("Save Configuration"):
            st.session_state.configs = configs
            st.success("Configuration saved successfully!")


if __name__ == "__main__":
    main()
