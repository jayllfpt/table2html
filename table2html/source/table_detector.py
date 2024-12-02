from ultralytics import YOLO

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


class TableDetector:
    def __init__(self, config: dict):
        self.model = YOLO(config["model_path"], task='detect')
        self.confidence_threshold = config.get("confidence_threshold", DEFAULT_CONF)
        self.iou_threshold = config.get("iou_threshold", DEFAULT_IOU)

    def __call__(self, image):
        """
        Detect table in the image
        Args:
            image: cv2 image in RGB format
        Returns:
            bbox: (x1, y1, x2, y2) or None if no table detected
        """
        results = self.model.predict(
            source=image, conf=self.confidence_threshold, iou=self.iou_threshold)
        if not results or len(results[0].boxes) == 0:
            return None
        # Return all detected table bboxes
        boxes = results[0].boxes
        return [
            (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3])
            )
            for box in boxes
        ]
