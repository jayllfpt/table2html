class TableProcessor:
    def __init__(self):
        pass

    def calculate_cells(self, rows, columns, image_shape, padding=2):
        """Calculate cell positions from rows and columns"""
        cells = []
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(columns):
                x1 = max(col[0] + padding, 0)
                y1 = max(row[1] + padding, 0)
                x2 = min(col[2] - padding, image_shape[1])
                y2 = min(row[3] - padding, image_shape[0])
                
                cells.append({
                    'row': row_idx,
                    'column': col_idx,
                    'box': (x1, y1, x2, y2)
                })
        return cells

    def assign_text_to_cells(self, cells, text_boxes, margin=2):
        """
        Assign recognized text to cells based on whether the center of the text box lies within the cell.

        Parameters:
        - cells: List of cell dictionaries with 'box'.
        - text_boxes: List of dictionaries with 'box', 'text', and 'center' keys.
        - margin: Margin to expand cell boundaries (default is 5 pixels).

        Returns:
        - Cells with added 'text' field containing OCR results.
        """
        for cell_idx, cell in enumerate(cells):
            x1_cell, y1_cell, x2_cell, y2_cell = cell['box']
            # Expand cell boundaries
            x1_cell -= margin
            y1_cell -= margin
            x2_cell += margin
            y2_cell += margin
            cell_texts = []

            for tb_idx, text_box in enumerate(text_boxes):
                x_center, y_center = text_box['center']
                text = text_box['text']

                # Check if the center lies within the expanded cell
                if (x1_cell <= x_center <= x2_cell) and (y1_cell <= y_center <= y2_cell):
                    cell_texts.append(text)

            # Join all texts assigned to this cell
            cell['text'] = ' '.join(cell_texts).strip()

        return cells