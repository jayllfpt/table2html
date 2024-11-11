import os
from .source import *

class Table2HTML:
    def __init__(self):
        # Initialize components
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.table_detector = TableDetector(model_path=os.path.join(current_dir, "models/det_table_v0.pt"))
        self.structure_detector = StructureDetector(
            row_model_path=os.path.join(current_dir, "models/det_row_v0.onnx"),
            column_model_path=os.path.join(current_dir, "models/det_col_v0.onnx")
        )
        self.ocr_engine = OCREngine()
        self.processor = TableProcessor()

    def __call__(self, image):
        """
        Convert a table image to HTML string
        
        Args:
            image: numpy.ndarray (OpenCV image)
            
        Returns:
            str: HTML table string or None if no table detected
        """
        # Detect table
        table_bbox = self.table_detector.detect(image)
        if table_bbox is None:
            return None

        # Detect rows and columns
        rows = self.structure_detector.detect_rows(image)
        columns = self.structure_detector.detect_columns(image)

        # Calculate cells
        cells = self.processor.calculate_cells(rows, columns, image.shape)
        
        # Perform OCR
        text_boxes = self.ocr_engine(image)
        
        # Assign text to cells
        self.cells = self.processor.assign_text_to_cells(cells, text_boxes)
        
        # Determine the number of rows and columns
        num_rows = max((cell['row'] for cell in self.cells), default=0) + 1
        num_cols = max((cell['column'] for cell in self.cells), default=0) + 1
        
        self.html = generate_html_table(self.cells, num_rows, num_cols)
        # Generate and return HTML table
        return self.cells, self.html