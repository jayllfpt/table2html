import os
from .source import *


class Table2HTML:
    def __init__(self):
        # Initialize components
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.table_detector = TableDetector(
            model_path=os.path.join(current_dir, "models/det_table_v0.pt"))
        self.structure_detector = StructureDetector(
            row_model_path=os.path.join(current_dir, "models/det_row_v0.pt"),
            column_model_path=os.path.join(current_dir, "models/det_col_v0.pt")
        )
        self.ocr_engine = OCREngine()
        self.processor = TableProcessor()

    def TableDetect(self, image):
        return {
            "table_bbox": self.table_detector(image)
        }

    def StructureDetect(self, table_image):
        # Detect rows and columns
        rows = self.structure_detector.detect_rows(table_image)
        columns = self.structure_detector.detect_columns(table_image)

        # Calculate cells
        cells = self.processor.calculate_cells(
            rows, columns, table_image.shape)

        # Perform OCR
        text_boxes = self.ocr_engine(table_image)

        # Assign text to cells
        cells = self.processor.assign_text_to_cells(cells, text_boxes)

        # Determine the number of rows and columns
        num_rows = max((cell['row'] for cell in cells), default=0) + 1
        num_cols = max((cell['column'] for cell in cells), default=0) + 1
        html = generate_html_table(cells, num_rows, num_cols)
        
        return {
            "cells": cells,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "html": html,
        }

    def __call__(self, image):
        """
        Convert a table image to HTML string

        Args:
            image: numpy.ndarray (OpenCV image)

        Returns:
            str: HTML table string or None if no table detected
        """
        extracted_data = self.TableDetect(image)
        if extracted_data["table_bbox"] is None:
            return None

        table_image = crop_image(image, extracted_data["table_bbox"])
        extracted_data.update(self.StructureDetect(table_image))

        return extracted_data
