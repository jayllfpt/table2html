from source.table_detector import TableDetector
from source.structure_detector import StructureDetector
from source.ocr_engine import OCREngine
from source.table_processor import TableProcessor
from source.utils import generate_html_table, load_image, visualize_boxes

class Table2HTML:
    def __init__(self):
        # Initialize components
        self.table_detector = TableDetector(model_path=r"models\det_table_v0.pt")
        self.structure_detector = StructureDetector(
            row_model_path=r'models\det_row_v0.onnx', 
            column_model_path=r'models\det_col_v0.onnx'
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
        cells = self.processor.assign_text_to_cells(cells, text_boxes)
        
        # Determine the number of rows and columns
        num_rows = max((cell['row'] for cell in cells), default=0) + 1
        num_cols = max((cell['column'] for cell in cells), default=0) + 1
        
        # Generate and return HTML table
        return generate_html_table(cells, num_rows, num_cols)

if __name__ == "__main__":
    image_path = r'images\sample.jpg'
    table2html = Table2HTML()
    html = table2html(load_image(image_path))
    print(html)
    # Save HTML table to file
    output_path = 'output_table.html'

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"HTML table saved to {output_path}")
