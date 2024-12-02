import os
from .source import *


class Table2HTML:
    def __init__(
        self,
        table_detection_model_path="",
        row_detection_model_path="",
        column_detection_model_path=""
    ):
        # Initialize components
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not len(table_detection_model_path):
            table_detection_model_path = os.path.join(
                current_dir, "models/det_table_v1.pt")
        if not len(row_detection_model_path):
            row_detection_model_path = os.path.join(
                current_dir, "models/det_row_v0.pt")
        if not len(column_detection_model_path):
            column_detection_model_path = os.path.join(
                current_dir, "models/det_col_v0.pt")

        self.table_detector = TableDetector(table_detection_model_path)
        self.structure_detector = StructureDetector(
            row_model_path=row_detection_model_path,
            column_model_path=column_detection_model_path
        )
        self.ocr_engine = OCREngine()
        self.processor = TableProcessor()

    def TableDetect(self, image):
        results = []
        tables = self.table_detector(image)
        for table in tables:
            results.append({
                "table_bbox": table,
            })
        return results

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

    def __call__(self, image, table_crop_padding=0):
        """
        Convert a table image to HTML string

        Args:
            image: numpy.ndarray (OpenCV image)

        Returns:
            str: HTML table string or None if no table detected
        """
        tables = self.TableDetect(image)
        if not len(tables):
            return None
        extracted_tables = []
        for table in tables:
            table_image = crop_image(
                image, table["table_bbox"], table_crop_padding)
            table.update(self.StructureDetect(table_image))
            extracted_tables.append(table)

        return extracted_tables
