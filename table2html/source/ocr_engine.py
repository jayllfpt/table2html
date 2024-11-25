from akaocr import BoxEngine, TextEngine
import numpy as np
import cv2


class OCREngine:
    def __init__(self, device="gpu"):
        self.box_engine = BoxEngine(device=device)
        self.text_engine = TextEngine(device=device)

    def _transform_image(self, image, box):
        """Transform perspective of text region"""
        # Get perspective transform image

        assert len(box) == 4, "Shape of points must be 4x2"
        img_crop_width = int(
            max(
                np.linalg.norm(box[0] - box[1]),
                np.linalg.norm(box[2] - box[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(box[0] - box[3]),
                np.linalg.norm(box[1] - box[2])))
        pts_std = np.float32([[0, 0],
                              [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(box, pts_std)
        dst_img = cv2.warpPerspective(
            image,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)

        img_height, img_width = dst_img.shape[0:2]
        if img_height/img_width >= 1.5:
            dst_img = np.rot90(dst_img, k=3)

        return dst_img

    def __call__(self, image):
        """
        Process image and return text boxes with recognized text
        Returns:
            List of dicts containing:
            - box: (x1, y1, x2, y2)
            - text: recognized text
            - center: (x_center, y_center)
        """
        boxes = self.box_engine(image)

        # Transform and recognize text for each box
        transformed_images = [self._transform_image(
            image, box) for box in boxes]
        texts = self.text_engine(transformed_images)
        texts = [text[0] for text in texts if text is not None]

        # Process results
        results = []
        for box, text in zip(boxes, texts):
            box_coords = np.array(box)
            x_coords = box_coords[:, 0]
            y_coords = box_coords[:, 1]
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))

            results.append({
                'box': (x1, y1, x2, y2),
                'text': text,
                'center': ((x1 + x2) / 2, (y1 + y2) / 2)
            })

        return results[::-1]
