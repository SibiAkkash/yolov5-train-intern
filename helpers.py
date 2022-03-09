import json
import cv2
from typing import Tuple

import secrets
import string
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
from pprint import pprint

from moviepy.editor import VideoFileClip
from pathlib import Path
import csv
import os


def get_time_elapsed_ms(start_frame: int, end_frame: int, fps: float):
    return 1000.0 * (end_frame - start_frame) / fps


def get_time_from_frame(frame_num: int, fps: float):
    return 1.0 * frame_num / fps


def get_random_string(length: int = 10, alphabet=string.ascii_letters + string.digits):
    return "".join([secrets.choice(alphabet) for _ in range(length)])


# image
# start_point: (X coordinate value, Y coordinate value).
# end_point:(X coordinate value, Y coordinate value).
# color
# thickness
def draw_line(img, start_pt, end_pt, color, thickness):
    pass


def is_bbox_inside_line(x1, y1, x2, y2, line_y: int) -> bool:
    return y1 >= line_y


def draw_text_with_box(
    img,
    text,
    base_coords,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1.5,
    thickness=2,
    padding=20,
    text_color=(0, 255, 0),
    box_color=(30, 30, 30),
):
    (w, h), baseline = cv2.getTextSize(
        text=text, fontFace=fontFace, fontScale=fontScale, thickness=thickness
    )

    text_pos_x, text_pos_y = list(map(int, base_coords))
    print(text_pos_x, text_pos_y)

    # background filled rect
    cv2.rectangle(
        img=img,
        pt1=(text_pos_x - padding, text_pos_y - h - padding),
        pt2=(text_pos_x + w + padding, text_pos_y + padding),
        color=box_color,
        thickness=-1,
    )

    # put text
    cv2.putText(
        img=img,
        text=text,
        org=(text_pos_x, text_pos_y),
        fontFace=fontFace,
        fontScale=fontScale,
        color=text_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def plot_to_img(fig, ax, num_cycles):

    labels = [
        "horn",
        "speedo",
        "exposed_fork",
        "torque_tool_hanging",
        "torque_tool_inserted",
        "ball_bearing_tool",
        "QR_code_scanner",
        "wheel_with_fender",
    ]

    color_dict = {
        "horn": "#55415f",
        "speedo": "#d77355",
        "exposed fork": "#646964",
        "torque_tool_hanging": "#508cd7",
        "torque_tool_inserted": "#64b964",
        "ball_bearing_tool": "#e6c86e",
        "QR_code_scanner": "#000000",
        "wheel_with_fender": "#9C0F48",
    }

    legend_elements = [Patch(facecolor=color_dict[i], label=i) for i in color_dict]

    if num_cycles == 1:
        plt.subplot(211)
        starts = [10.5, 2.56, 2.56, 2.56, 25.7, 10.56, 30.57, 1.2]
        start_to_end_num = [22.3, 48.9, 48.9, 48.9, 10.4, 5, 10, 22]
        ax[0].barh(labels, start_to_end_num, left=starts, color=color_dict.values())
        ax[0].set_title("cycle 1")
        ax[0].set_xlabel("Time in seconds")
        # plt.xlabel("time in seconds")

    if num_cycles == 2:
        # cycle 1
        plt.subplot(211)
        starts = [10.5, 2.56, 2.56, 2.56, 25.7, 10.56, 30.57, 1.2]
        start_to_end_num = [22.3, 48.9, 48.9, 48.9, 10.4, 5, 10, 22]
        ax[0].barh(labels, start_to_end_num, left=starts, color=color_dict.values())
        ax[0].set_title("cycle 1")

        plt.subplot(212)
        starts = [58, 48.24, 48.24, 59.44, 56.24, 48.24, 137.84, 47.56]
        start_to_end_num = [50.2, 120.2, 129.23, 76.12, 35.8, 39.64, 35, 129.92]
        ax[1].barh(labels, start_to_end_num, left=starts, color=color_dict.values())
        ax[1].set_title("cycle 2")
        ax[1].set_xlabel("Time in seconds")
        # plt.xlabel("time in seconds")

    plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


labels = [
    "horn",
    "speedo",
    "exposed_fork",
    "torque_tool_hanging",
    "torque_tool_inserted",
    "ball_bearing_tool",
    "QR_code_scanner",
    "wheel_with_fender",
]

colors = [
    "#55415f",
    "#d77355",
    "#646964",
    "#508cd7",
    "#64b964",
    "#e6c86e",
    "#000000",
    "#9C0F48",
]


def plot_last_2_cycles(fig, ax, cycles):
    for i, cycle in enumerate(cycles):
        plt.subplot(int(f"21{i+1}"))

        cycle_start_t = cycle["cycle_start_frame_time"]
        starts = []
        elapsed = []

        # for obj_id, (first_seen_t, last_seen_t) in cycle["marker_frame_times"].items():
        #     start_t = first_seen_t if first_seen_t > 0 else 0
        #     starts.append(start_t)
        #     t_elapsed = round(last_seen_t - first_seen_t, 2) if last_seen_t > 0 else 0
        #     elapsed.append(t_elapsed)

        # # shift time to cycle start
        # for idx, t in enumerate(starts):
        #     starts[idx] = max(0, t - cycle_start_t)

        for obj_id, (first_seen_rt, last_seen_rt) in cycle[
            "marker_times_relative"
        ].items():
            start_t = round(first_seen_rt, 2) if first_seen_rt > 0 else 0
            starts.append(start_t)
            t_elapsed = (
                round(last_seen_rt - first_seen_rt, 2) if last_seen_rt > 0 else 0
            )
            elapsed.append(t_elapsed)

        # print(f'{starts = }')
        # print(f'{elapsed = }')

        ax[i].cla()
        ax[i].barh(labels, elapsed, left=starts, color=colors)
        ax[i].set_title(f"cycle {cycle['cycle_num']}")
        ax[i].set_xlabel("Time in seconds")

        # ticks
        xticks = np.arange(0, 100, 5)
        ax[i].set_xticks(xticks)

        ax[i].set_axisbelow(True)
        ax[i].xaxis.grid(color="gray", linestyle="dashed")

    plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def plot():
    fig, ax = plt.subplots(1, figsize=(16, 6))

    labels = [
        "horn",
        "speedo",
        "exposed_fork",
        "torque_tool_hanging",
        "torque_tool_inserted",
        "ball_bearing_tool",
        "QR_code_scanner",
        "wheel_with_fender",
    ]

    starts = [10.5, 2.56, 2.56, 2.56, 25.7, 3.56, 30.57, 1.2]
    start_to_end_num = [12.3, 48.9, 48.9, 48.9, 35.8, 35, 35, 12]

    cyc_start = 1188
    cyc_end = 4437

    st = np.array([0, 0, 1206, 1486, 1406, 0, 3446, 1189])
    ends = np.array([4437, 4437, 4437, 4437, 3184, 4437, 4432, 4437])
    st_ends = ends - st

    st_2 = np.array([])

    color_dict = {
        "horn": "#55415f",
        "speedo": "#d77355",
        "tool hanging": "#508cd7",
        "tool inserted": "#64b964",
        "ball bearing tool": "#e6c86e",
        "QR code scanner": "#000000",
        "wheel with fender": "#9C0F48",
        "exposed fork": "#646964",
    }

    legend_elements = [Patch(facecolor=color_dict[i], label=i) for i in color_dict]

    # plt.legend(handles=legend_elements)

    assert len(labels) == len(starts) == len(start_to_end_num)

    ax.barh(labels, start_to_end_num, left=starts, color=color_dict.values())
    # ax.barh(labels, st_ends, left=st, color=color_dict.values())

    # xticks = np.arange(cyc_start - 10, cyc_end + 10, 3)
    # ax.set_xticks(xticks)

    plt.show()


def plot_global_cycles(file):
    matplotlib.use("TkAgg")
    with open(file) as f:
        times = list(map(lambda t: float(t[:-1]), f.readlines()))

    fig, ax = plt.subplots(1, figsize=(16, 6))

    cycles = range(0, len(times))
    color = "#1f7ef280"

    # ax.bar(cycles, times, color=color)
    # ax.plot(cycles, times, color=color)
    ax.scatter(cycles, times)

    # xticks = np.arange(0, len(times), 1)
    # ax.set_xticks(xticks)
    yticks = np.arange(0, max(times), 10)
    ax.set_yticks(yticks)

    ax.set_title("Global cycle times")
    # ax.set_xlabel("Cycle number")
    ax.set_ylabel("Cycle time (sec)")

    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")

    y_mean = [np.median(times)] * len(times)

    print(f"median: {np.median(times)}")

    ax.plot(cycles, y_mean, label="Mean", linestyle="dashed", color="green")

    plt.show()


def plot_bbox_sizes(file):

    data = pd.read_csv(file)
    print(data["scooter_id"].unique())

    for id in data["scooter_id"].unique():
        if id in range(50):
            scooter_data = data[data["scooter_id"] == id]
            plt.scatter(
                scooter_data["w"],
                scooter_data["h"],
                label=f"Scooter {id}",
                s=[2] * scooter_data.shape[0],
            )

    ys = np.arange(0, 900, 10)
    xs = [400] * ys.shape[0]
    plt.plot(xs, ys, linestyle="dashed", color="green")

    xs = np.arange(0, 900, 10)
    ys = [800] * xs.shape[0]
    plt.plot(xs, ys, linestyle="dashed", color="green")

    plt.xticks(np.arange(0, 1000, 100))
    plt.yticks(np.arange(0, 1000, 100))

    plt.legend()
    plt.show()


def get_vid_clip(path):
    start = 10.56
    end = 14.28
    orig_video = VideoFileClip(path)
    clip = orig_video.subclip(start, end)
    clip.write_videofile("/home/sibi/Downloads/cycle_videos/rec_3_clip.mp4")

    clip.close()
    orig_video.close()


def get_action_clips(data_root=Path("."), save_root=Path("../action_clips")):
    csv_file_path = "crop_videos/data.csv"
    data = pd.read_csv(csv_file_path)
    print(data)
    action_ids = data["action_id"].unique()

    with open("crop_videos/classnames.txt") as f:
        classnames = list(map(lambda c: c.strip(), f.readlines()))

    print(action_ids)
    print(classnames)

    for action_id in action_ids:
        print(f"creating directory {classnames[action_id]}")
        os.makedirs(save_root / classnames[action_id], exist_ok=True)

    num_actions = [0] * len(classnames)

    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # skip header

        for row in csv_reader:
            if not row:
                print("empty row")
                continue

            print(row)
            vid_path, start, end, action_id = row
            action_id = int(action_id)
            num_actions[action_id] += 1

            clip = VideoFileClip(str(data_root / vid_path)).subclip(start, end)
            clip_path = (
                save_root / classnames[action_id] / f"{num_actions[action_id]}.mp4"
            )
            clip.write_videofile(str(clip_path))

            clip.close()


def convert_to_millis(s):
    hh, mm, ss_ms = s.split(":")
    ss, ms = ss_ms.split(".")
    hh, mm, ss, ms = int(hh), int(mm), int(ss), int(ms)
    total_ms = ms + (ss * 1000) + (mm * 60 * 1000) + (hh * 24 * 60 * 1000)
    return total_ms


def plot_action_durations():
    import matplotlib.transforms as mtransforms

    data = pd.read_csv("crop_videos/data.csv")

    with open("crop_videos/classnames.txt") as f:
        classnames = list(map(lambda c: c.strip(), f.readlines()))

    fig, axs = plt.subplot_mosaic(
        """
        0011
        0011
        2233
        2233
        4455
        4455
        """,
        constrained_layout=True,
        figsize=(200, 100)
    )

    for action_id in data["action_id"].unique():
        actions = data[data["action_id"] == action_id]

        starts = actions["start_time"].to_numpy()
        ends = actions["end_time"].to_numpy()

        starts_ms = np.array(list(map(lambda t: convert_to_millis(t), starts)))
        ends_ms = np.array(list(map(lambda t: convert_to_millis(t), ends)))
        diff = (ends_ms - starts_ms) / 1000

        for st, et, dt in zip(starts, ends, diff):
            if dt < 0:
                print(st, et, dt)

        axs[str(action_id)].plot(np.arange(0, len(starts), 1), diff)

        # median
        # median_diff = np.median(diff)
        # axs[str(action_id)].plot(
        #     np.arange(0, len(starts), 1),
        #     [median_diff] * len(starts),
        #     linestyle="dashed",
        #     color="green",
        # )

        axs[str(action_id)].set_xticks(np.arange(0, 50, 5))
        axs[str(action_id)].set_yticks(np.arange(0, 15, 1))

        axs[str(action_id)].set_title(classnames[action_id])

    plt.show()


if __name__ == "__main__":
    # plot_global_cycles(file='cycle_times/cycle_times_wheel.txt')
    # plot_bbox_sizes(file="cycle_times/bbox_sizes.csv")
    # get_vid_clip("/home/sibi/Downloads/cycle_videos/rec_3.mp4")
    # get_action_clips()
    plot_action_durations()
