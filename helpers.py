import cv2
from typing import Tuple

import secrets
import string

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd



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


def plot():
    fig, ax = plt.subplots(1, figsize=(16, 6))

    labels = [
        'horn', 
        'speedo', 
        'exposed_fork', 
        'torque_tool_hanging', 
        'torque_tool_inserted', 
        'ball_bearing_tool', 
        'QR_code_scanner', 
        'wheel_with_fender'
    ]

    starts = [10.5, 2.56, 2.56, 2.56, 25.7, 3.56, 30.57, 1.2]
    start_to_end_num = [12.3, 48.9, 48.9, 48.9, 35.8, 35, 35, 12]

    # 0: (-1, -1),
    # 1: (-1, -1),
    # 2: (1206, 4437),
    # 3: (1486, 4437),
    # 4: (1406, 3184),
    # 5: (-1, -1),
    # 6: (3446, 4432),
    # 7: (1189, 4437)

    # exposed_fork: 
    # first_seen: 48.24, 
    # last_seen: 177.48, 

    # torque_tool_hanging: 
    # first_seen: 59.44, 
    # last_seen: 177.48, 

    # torque_tool_inserted: 
    # first_seen: 56.24, 
    # last_seen: 127.36, 

    # ball_bearing_tool: 
    # first_seen: -0.04, 
    # last_seen: -0.04, 

    # QR_code_scanner: 
    # first_seen: 137.84, 
    # last_seen: 177.28, 

    # wheel_with_fender: 
    # first_seen: 47.56, 
    # last_seen: 177.48, 


    cyc_start = 1188
    cyc_end = 4437

    st = np.array([0, 0, 1206, 1486, 1406, 0, 3446, 1189])
    ends = np.array([4437, 4437, 4437, 4437, 3184, 4437, 4432, 4437])
    st_ends = ends - st

    st_2 = np.array([ ])

    color_dict = {
        'horn':'#55415f',
        'speedo':'#d77355',
        'tool hanging':'#508cd7',
        'tool inserted':'#64b964',
        'ball bearing tool':'#e6c86e',
        'QR code scanner': '#000000',
        'wheel with fender': '#9C0F48',
        'exposed fork': '#646964'
    }

    legend_elements = [
        Patch(facecolor=color_dict[i], label=i)  for i in color_dict
    ]
    
    # plt.legend(handles=legend_elements)

    assert(len(labels) == len(starts) == len(start_to_end_num))
    
    ax.barh(labels, start_to_end_num, left=starts, color=color_dict.values())
    # ax.barh(labels, st_ends, left=st, color=color_dict.values())


    # xticks = np.arange(cyc_start - 10, cyc_end + 10, 3)
    # ax.set_xticks(xticks)

    plt.show()

if __name__ == "__main__":
    plot()



# {0: (-1, -1),
# 1: (-1, -1),
# 2: (345, 1685),
# 3: (345, 1685),
# 4: (269, 1660),
# 5: (327, 328),
# 6: (317, 1200),
# 7: (269, 1685)}}
