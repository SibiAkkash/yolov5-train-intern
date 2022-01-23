import cv2
from typing import Tuple

# image
# start_point: (X coordinate value, Y coordinate value).
# end_point:(X coordinate value, Y coordinate value).
# color
# thickness

def draw_line(img, start_pt, end_pt, color, thickness):
    pass

def is_bbox_inside_line(x1, y1, x2, y2, line_y: int) -> bool:
    return y1 >= line_y

