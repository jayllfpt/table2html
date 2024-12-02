import unittest
import cv2
import numpy as np
from table2html import Table2HTML

class TestTable2HTML(unittest.TestCase):

    def setUp(self):
        # Set up configurations and initialize Table2HTML
        table_config = {
            "model_path": "table2html/models/det_table_v1.pt",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
        }
        row_config = {
            "model_path": "table2html/models/det_row_v0.pt",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "task": "detect",
        }
        column_config = {
            "model_path": "table2html/models/det_col_v0.pt",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "task": "detect",
        }
        self.table2html = Table2HTML(table_config, row_config, column_config)

    def test_table_detection(self):
        # Load a sample image
        image = cv2.imread("table2html/images/sample.jpg")
        self.assertIsNotNone(image, "Failed to load image")

        # Test table detection
        detection_data = self.table2html.TableDetect(image)
        self.assertIsInstance(detection_data, list, "Detection data should be a list")
        self.assertGreater(len(detection_data), 0, "No tables detected")

    def test_structure_detection(self):
        # Load a sample image
        image = cv2.imread("table2html/images/sample.jpg")
        self.assertIsNotNone(image, "Failed to load image")

        # Test structure detection
        data = self.table2html.StructureDetect(image)
        self.assertIsInstance(data, dict, "Structure data should be a dictionary")
        self.assertIn("html", data, "HTML output missing from structure data")

    def test_full_pipeline(self):
        # Load a sample image
        image = cv2.imread("table2html/images/sample.jpg")
        self.assertIsNotNone(image, "Failed to load image")

        # Test full pipeline
        detection_data = self.table2html(image)
        self.assertIsInstance(detection_data, list, "Detection data should be a list")
        self.assertGreater(len(detection_data), 0, "No tables detected in full pipeline")

if __name__ == '__main__':
    unittest.main() 