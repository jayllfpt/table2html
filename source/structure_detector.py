from ultralytics import YOLO
import numpy as np

class StructureDetector:
    def __init__(self, row_model_path, column_model_path):
        self.row_model = YOLO(row_model_path, task='detect')
        self.column_model = YOLO(column_model_path, task='detect')

    def _detect_elements(self, image, model):
        """Helper method to detect elements using specified model"""
        results = model.predict(source=image)
        if not results:
            return []
        return [(int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                 int(box.xyxy[0][2]), int(box.xyxy[0][3])) 
                for box in results[0].boxes]

    def detect_rows(self, table_image):
        """Detect and return sorted row bboxes"""
        rows = self._detect_elements(table_image, self.row_model)
        return sorted(rows, key=lambda box: box[1])  # Sort by y-coordinate

    def detect_columns(self, table_image):
        """Detect and return sorted column bboxes"""
        columns = self._detect_elements(table_image, self.column_model)
        return sorted(columns, key=lambda box: box[0])  # Sort by x-coordinate 