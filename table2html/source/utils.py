import cv2

def crop_image(image, bbox, padding=0):
    """Crop image to bounding box"""
    return image[bbox[1]-padding:bbox[3]+padding, bbox[0]-padding:bbox[2]+padding]

def visualize_boxes(image, boxes, color = (0, 0, 255), thickness = 1):
    """
    Visualize bounding boxes on the image.

    Parameters:
    - image: The image on which to draw.
    - boxes: List of bounding boxes as (x1, y1, x2, y2).
    - color: Color of the bounding box edges (BGR format).
    - title: Title of the plot.

    Returns:
    - Image with drawn bounding boxes (BGR format)
    """
    # Create a copy of the image to avoid modifying the original
    vis_image = image.copy()

    # Draw boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

    return vis_image


def generate_html_table(cells, num_rows, num_cols):
    """
    Generate HTML table based on detected cells and their content.

    Parameters:
    - cells: List of cell dictionaries with 'row', 'column', and 'text'.
    - num_rows: Total number of rows in the table.
    - num_cols: Total number of columns in the table.

    Returns:
    - A string containing the HTML representation of the table.
    """
    # Initialize the table with empty strings
    table_data = [['' for _ in range(num_cols)] for _ in range(num_rows)]

    # Map each cell's text to the correct table position
    # print("Mapping Cells to Rows and Columns:")
    for cell in cells:
        row = cell['row']
        col = cell['column']
        text = cell.get('text', '')
        # print(f"Cell at (row={row}, col={col}): {text}")
        table_data[row][col] = text

    # Build the HTML table
    html = '<table border="1" style="border-collapse: collapse; width: 100%;">\n'
    for i, row_data in enumerate(table_data):
        html += '  <tr>\n'
        for cell_text in row_data:
            if i == 0:
                # Header row
                html += f'    <th style="padding: 8px; background-color: #f2f2f2;">{cell_text}</th>\n'
            else:
                # Data rows
                html += f'    <td style="padding: 8px;">{cell_text}</td>\n'
        html += '  </tr>\n'
    html += '</table>'
    return html

def load_image(image_path):
    """Load and convert image to RGB"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 