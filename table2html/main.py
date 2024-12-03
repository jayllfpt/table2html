import os
from .source import *

DEFAULT_TABLE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/table_detection.pt")
DEFAULT_ROW_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/row_detection.pt")
DEFAULT_COLUMN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/column_detection.pt")

class Table2HTML:
    def __init__(
        self,
        table_detection_config: dict,
        row_detection_config: dict,
        column_detection_config: dict
    ):
        # Initialize components
        # Set default model paths if not provided in configs
        if "model_path" not in table_detection_config or not table_detection_config["model_path"]:
            table_detection_config["model_path"] = DEFAULT_TABLE_MODEL_PATH
        if "model_path" not in row_detection_config or not row_detection_config["model_path"]:
            row_detection_config["model_path"] = DEFAULT_ROW_MODEL_PATH
        if "model_path" not in column_detection_config or not column_detection_config["model_path"]:
            column_detection_config["model_path"] = DEFAULT_COLUMN_MODEL_PATH

        self.table_detector = TableDetector(table_detection_config)
        self.structure_detector = StructureDetector(
            row_config=row_detection_config,
            column_config=column_detection_config
        )
        self.ocr_engine = OCREngine()
        self.processor = TableProcessor()

    def TableDetect(self, image):
        tables = self.table_detector(image)
        if len(tables):
            return [
                {
                "table_bbox": table,
            }
                for table in tables
            ]
        return []

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
            return []
        extracted_tables = []
        for table in tables:
            table_image = crop_image(
                image, table["table_bbox"], table_crop_padding)
            table.update(self.StructureDetect(table_image))
            extracted_tables.append(table)

        return extracted_tables
