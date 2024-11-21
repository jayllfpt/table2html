from ultralytics import YOLO

class TableDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='detect')
    
    def __call__(self, image):
        """
        Detect table in the image
        Args:
            image: cv2 image in RGB format
        Returns:
            bbox: (x1, y1, x2, y2) or None if no table detected
        """
        results = self.model.predict(source=image)
        if not results or len(results[0].boxes) == 0:
            return None
        # Return the first detected table bbox
        box = results[0].boxes[0]
        return (
            int(box.xyxy[0][0]), 
            int(box.xyxy[0][1]),
            int(box.xyxy[0][2]), 
            int(box.xyxy[0][3])
        ) 