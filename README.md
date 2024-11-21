# Table2HTML

A Python package that converts table images into HTML format using Object Detection model and OCR.

## Installation

```bash
pip install table2html
```

## Usage

### Initialize
```python
from table2html import Table2HTML
table2html = Table2HTML()
```

### Table Detection
```python
detection_data = table2html.TableDetect(image)
# Output: {"table_bbox": Tuple[int]}

# Visualize table detection
from table2html.source import visualize_boxes
cv2.imwrite(
    "table_detection.jpg", 
    visualize_boxes(
        image, 
        [detection_data["table_bbox"]], 
        color=(0, 0, 255),
        thickness=1
    )
)
```
Table detection result:

![Table Detection Example](table2html\images\table_detection.jpg)

### Structure Detection
```python
data = table2html.StructureDetect(image)
# Output: {
#   "cells": List[Dict],
#   "num_rows": int,
#   "num_cols": int,
#   "html": str
# }

# Visualize structure detection
from table2html.source import visualize_boxes
cv2.imwrite(
    "structure_detection.jpg", 
    visualize_boxes(
        image, 
        [cell['box'] for cell in data['cells']], 
        color=(0, 255, 0),
        thickness=1
    )
)

# Write HTML output
with open('table.html', 'w') as f:
    f.write(data["html"])
```

Structure detection result:

![Structure Detection Example](table2html\images\table_cells.jpg)

### Full Pipeline
**Note:** The cell coordinates are relative to the cropped table image.
```python
data = table2html(image)
# Output: {
#   "table_bbox": Tuple[int],
#   "cells": List[Dict],
#   "num_rows": int,
#   "num_cols": int,
#   "html": str
# } 
```

Extracted html:
| Age | Sample Size | Mean | S.E. | Ratio Index a |
|-----|-------------|------|------|---------------|
| 19-30years | 5 | 78980 | 15580 | 5.1 |
| 31-35years | 13 | 25300 | 4860 | 2.1 |
| 36-40 years | 14 | 21450 | 2650 | 4.7 |
| 41-45 years | 32 | 7320 | 1450 | 0.8 |
| >45 years | 36 | 1880 | 310 | 1.0 |


## Input
- `image`: numpy.ndarray (OpenCV/cv2 image format)

## Outputs
1. `table_bbox`: Tuple[int] - Bounding box coordinates (x1, y1, x2, y2) of the table
2. `cells`: List[Dict] - List of cell dictionaries, where each dictionary contains:
   - `row`: int - Row index
   - `column`: int - Column index
   - `box`: Tuple[int] - Bounding box coordinates (x1, y1, x2, y2)
   - `text`: str - Cell text content
3. `num_rows`: int - Number of rows in the table
4. `num_cols`: int - Number of columns in the table
5. `html`: str - HTML representation of the table

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
