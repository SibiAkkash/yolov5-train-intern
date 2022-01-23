import cv2
from typing import Tuple

import secrets
import string

import matplotlib.pyplot as plt


def get_time_elapsed_ms(start_frame: int, end_frame: int, fps: float):
    return 1000.0 * (end_frame - start_frame) / fps

def get_time_from_frame(frame_num: int, fps: float):
    return 1.0 * frame_num / fps
    
def get_random_string(length: int = 10, alphabet = string.ascii_letters + string.digits):
    return ''.join([secrets.choice(alphabet) for _ in range(length)])

# image
# start_point: (X coordinate value, Y coordinate value).
# end_point:(X coordinate value, Y coordinate value).
# color
# thickness
def draw_line(img, start_pt, end_pt, color, thickness):
    pass

def is_bbox_inside_line(x1, y1, x2, y2, line_y: int) -> bool:
    return y1 >= line_y

