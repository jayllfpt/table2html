from table2html.source import visualize_boxes
from table2html.source import crop_image
from table2html import Table2HTML
import cv2


table_crop_padding = 15

table_config = {

}

row_config = {

}

column_config = {

}

table2html = Table2HTML(table_config, row_config, column_config)

image = cv2.imread(
    r"C:\Users\jayll\Downloads\no_tables.jpg")

detection_data = table2html(image, table_crop_padding)

for i, data in enumerate(detection_data):
    table_image = crop_image(image, data["table_bbox"], table_crop_padding)
    cv2.imwrite(
        "table_detection.jpg",
        visualize_boxes(
            image,
            [data["table_bbox"]],
            color=(0, 0, 255),
            thickness=1
        )
    )
    cv2.imwrite(
        "structure_detection.jpg",
        visualize_boxes(
            table_image,
            [cell['box'] for cell in data['cells']],
            color=(0, 255, 0),
            thickness=1
        )
    )

    with open(f"table_{i}.html", "w") as f:
        f.write(data["html"])