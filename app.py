import cv2
import streamlit as st
from table2html import Table2HTML
import numpy as np
import time


def main():
    st.title("Image to HTML Table Converter")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert uploaded file to opencv image
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image',
                 use_column_width=True)

        # Start timer
        start_time = time.time()

        # Convert to HTML table
        table2html = Table2HTML()
        cells, html = table2html(image)

        # Calculate execution time
        execution_time = time.time() - start_time
        st.info(f"Conversion completed in {execution_time:.2f} seconds")

        st.subheader("HTML Table:")
        st.markdown(html, unsafe_allow_html=True)

        # Download button for HTML
        st.download_button(
            label="Download HTML",
            data=html,
            file_name="table.html",
            mime="text/html"
        )


if __name__ == "__main__":
    main()
