# Table2HTML

A Python package that converts table images into HTML format using Object Detection model and OCR.

## Installation

```bash
pip install table2html
```

## Usage

```python
from table2html import Table2HTML
import cv2

# Initialize
table2html = Table2HTML()

# Convert table image to HTML
image = cv2.imread("path/to/image.jpg")
cells, html = table2html(image)

# Save HTML output
with open('table.html', 'w') as f:
    f.write(html)
```

## Input
- `image`: numpy.ndarray (OpenCV/cv2 image format)

## Outputs
1. `cells`: List of dictionaries, where each dictionary represents a cell with:
   - `row`: int - Row index
   - `column`: int - Column index
   - `box`: Tuple[int] - Bounding box coordinates (x1, y1, x2, y2)
   - `text`: str - Cell text content

2. `html`: str - HTML representation of the table

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
