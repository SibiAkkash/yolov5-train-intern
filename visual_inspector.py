from __future__ import annotations
from typing import Dict, List, Tuple

# from helpers import get_time_elapsed_ms, get_random_string
from pprint import pprint
import json
import numpy as np
import numpy.typing as npt
from pathlib import Path
import cv2
import mysql.connector


def is_object_present(detections: List[int], object_id: int):
    return object_id in detections


# ----------------- FLOW -------------------------------------------------------------------------------------
# start of cycle is speedo crossing top line
# if horn crosses bottom line, the cycle has finished, refresh state
# store first and last occurence of bboxes
# ------------------------------------------------------------------------------------------------------------


class VisualInspector:
    def __init__(
        self,
        start_marker_object_id: int,
        end_marker_object_id: int,
        # process_object_ids: List[int],
        stream_fps: float,
        entry_line_y: int,
        exit_line_y: int,
        marker_names: List[str],
    ):
        self.start_marker_object_id = start_marker_object_id
        self.end_marker_object_id = end_marker_object_id
        # self.process_object_ids = process_object_ids
        self.marker_names = marker_names

        # self.NUM_STEPS = len(process_object_ids)

        self.stream_fps = stream_fps

        self.entry_line_y = entry_line_y
        self.exit_line_y = exit_line_y

        self.refresh_state()

    def refresh_state(self) -> None:

        self.state = {
            "cycle_started": False,
            "cycle_ended": False,
            "cycle_start_frame_num": -1,
            "cycle_end_frame_num": -1,
            "marker_frame_numbers": {
                i: (-1, -1) for i in range(len(self.marker_names))
            },
        }

    def cycle_started(self) -> bool:
        return self.state["cycle_started"]

    def _handle_cycle_start(self, frame_num: int):
        self.state["cycle_started"] = True
        self.state["cycle_start_frame_num"] = frame_num

        # self.state["seen_objects"].append(self.start_marker_object_id)

        print("CYCLE STARTED")
        pprint(self.state)

    def _handle_cycle_end(self, frame_num: int):
        # if there was any step, do necessary processing

        self.refresh_state()
        print("CYCLE ENDED")
        pprint(self.state)

    def process_detections(
        self,
        detections: Dict[int, Tuple[int, int, int, int]],
        frame_num: int,
        current_frame: npt.NDArray = None,
    ):
        print(frame_num)
        pprint(detections)


# def show_steps(image):
#     # box background
#     cv2.rectangle(
#         img=image,
#         color=(100, 100, 100),
#         pt1=RECT_POINT_1,
#         pt2=RECT_POINT_2,
#         thickness=-1,
#     )
#     # steps status
#     for step in range(NUM_STEPS + 1):
#         if step == NUM_STEPS:
#             s = f"Sequence: {state['sequence']}"
#             color = DONE_COLOUR if state["cycle_ended"] else NOT_DONE_COLOUR
#         else:
#             s = f"Step {step + 1}, time taken: {state['step_times'][step]} ms"
#             color = DONE_COLOUR if state["is_step_completed"][step] else NOT_DONE_COLOUR

#         cv2.putText(
#             img=image,
#             text=s,
#             org=(X_OFFSET, Y_OFFSET + Y_PADDING * step),
#             fontFace=0,
#             fontScale=0.5,
#             color=color,
#             thickness=1,
#         )
