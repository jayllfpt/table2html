from ultralytics import YOLO
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


class StructureDetector:
    def __init__(self, row_config: dict, column_config: dict):
        self.row_config = row_config
        self.column_config = column_config
        self.row_model = YOLO(
            row_config["model_path"], task=row_config.get("task", "detect"))
        self.column_model = YOLO(
            column_config["model_path"], task=column_config.get("task", "detect"))

    def _detect_elements(self, image, model, conf=DEFAULT_CONF, iou=DEFAULT_IOU):
        """Helper method to detect elements using specified model"""
        results = model.predict(source=image, conf=conf, iou=iou)
        if not results:
            return []
        return [(int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                 int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                for box in results[0].boxes]

    def _merge_overlapping_rows(self, rows, overlap_threshold=0.5):
        """
        Merge overlapping row boxes based on IoU threshold.
        Args:
            rows: List of row bboxes (x1,y1,x2,y2)
            overlap_threshold: IoU threshold for merging
        Returns:
            List of merged row bboxes
        """
        if not rows:
            return []

        # Sort by y coordinate
        rows = sorted(rows, key=lambda box: box[1])
        merged = []
        current = rows[0]

        for next_box in rows[1:]:
            # Calculate overlap
            y_top = max(current[1], next_box[1])
            y_bottom = min(current[3], next_box[3])
            overlap = max(0, y_bottom - y_top)

            # Calculate heights
            h1 = current[3] - current[1]
            h2 = next_box[3] - next_box[1]

            # If significant overlap, merge boxes
            if overlap > overlap_threshold * min(h1, h2):
                # Merge into taller box
                current = (
                    min(current[0], next_box[0]),  # x1
                    min(current[1], next_box[1]),  # y1
                    max(current[2], next_box[2]),  # x2
                    max(current[3], next_box[3])   # y2
                )
            else:
                merged.append(current)
                current = next_box

        merged.append(current)
        return merged

    def _merge_overlapping_columns(self, columns, overlap_threshold=0.5):
        """
        Merge overlapping column boxes based on IoU threshold.
        Args:
            columns: List of column bboxes (x1,y1,x2,y2)
            overlap_threshold: IoU threshold for merging
        Returns:
            List of merged column bboxes
        """
        if not columns:
            return []

        # Sort by x coordinate
        columns = sorted(columns, key=lambda box: box[0])
        merged = []
        current = columns[0]

        for next_box in columns[1:]:
            # Calculate overlap
            x_left = max(current[0], next_box[0])
            x_right = min(current[2], next_box[2])
            overlap = max(0, x_right - x_left)

            # Calculate widths
            w1 = current[2] - current[0]
            w2 = next_box[2] - next_box[0]

            # If significant overlap, merge boxes
            if overlap > overlap_threshold * min(w1, w2):
                # Merge into wider box
                current = (
                    min(current[0], next_box[0]),  # x1
                    min(current[1], next_box[1]),  # y1
                    max(current[2], next_box[2]),  # x2
                    max(current[3], next_box[3])   # y2
                )
            else:
                merged.append(current)
                current = next_box

        merged.append(current)
        return merged

    def detect_rows(self, table_image):
        """Detect and return sorted row bboxes"""
        rows = self._detect_elements(
            table_image,
            self.row_model,
            conf=self.row_config.get("confidence_threshold", DEFAULT_CONF),
            iou=self.row_config.get("iou_threshold", DEFAULT_IOU)
        )
        rows = self._merge_overlapping_rows(rows)
        return sorted(rows, key=lambda box: box[1])  # Sort by y-coordinate

    def detect_columns(self, table_image):
        """Detect and return sorted column bboxes"""
        columns = self._detect_elements(
            table_image,
            self.column_model,
            conf=self.column_config.get("confidence_threshold", DEFAULT_CONF),
            iou=self.column_config.get("iou_threshold", DEFAULT_IOU)
        )
        columns = self._merge_overlapping_columns(columns)
        return sorted(columns, key=lambda box: box[0])  # Sort by x-coordinate
