from ultralytics import YOLO
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


class StructureDetector:
    def __init__(self, row_config: dict, column_config: dict):
        self.row_config = row_config
        self.column_config = column_config
        self.row_model = YOLO(row_config["model_path"], task=row_config.get("task", "detect"))
        self.column_model = YOLO(column_config["model_path"], task=column_config.get("task", "detect"))

    def _detect_elements(self, image, model, conf=DEFAULT_CONF, iou=DEFAULT_IOU):
        """Helper method to detect elements using specified model"""
        results = model.predict(source=image, conf=conf, iou=iou)
        if not results:
            return []
        return [(int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                 int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                for box in results[0].boxes]

    def detect_rows(self, table_image):
        """Detect and return sorted row bboxes"""
        rows = self._detect_elements(
            table_image, 
            self.row_model, 
            conf=self.row_config.get("confidence_threshold", DEFAULT_CONF), 
            iou=self.row_config.get("iou_threshold", DEFAULT_IOU)
        )
        return sorted(rows, key=lambda box: box[1])  # Sort by y-coordinate

    def detect_columns(self, table_image):
        """Detect and return sorted column bboxes"""
        columns = self._detect_elements(
            table_image, 
            self.column_model, 
            conf=self.column_config.get("confidence_threshold", DEFAULT_CONF), 
            iou=self.column_config.get("iou_threshold", DEFAULT_IOU)
        )
        return sorted(columns, key=lambda box: box[0])  # Sort by x-coordinate
