from __future__ import annotations
from tabnanny import check

from typing import Dict, List, Tuple

# from helpers import get_time_elapsed_ms, get_random_string
from pprint import pprint
import json
import numpy as np
import numpy.typing as npt
from pathlib import Path
import cv2

from dataclasses import dataclass, field

from helpers import get_time_elapsed_ms, get_time_from_frame


def is_object_present(detections: List[int], object_id: int):
    return object_id in detections


# ----------------- FLOW -------------------------------------------------------------------------------------
# start of cycle is speedo crossing top line
# if horn crosses bottom line, the cycle has finished, refresh state
# store first and last occurence of bboxes
# ------------------------------------------------------------------------------------------------------------

@dataclass
class Cycle:
    start_frame_num: int
    end_frame_num: int
    started: bool
    ended: bool
    marker_frame_numbers: dict = field(default_factory=dict)


class VisualInspector:
    def __init__(self, 
        start_marker_object_id: int,
        end_marker_object_id: int,
        stream_fps: float,
        entry_line_y: int,
        exit_line_y: int,
        marker_names: List[str],
    ):
        self.start_marker_object_id = start_marker_object_id
        self.end_marker_object_id = end_marker_object_id
        self.stream_fps = stream_fps
        self.entry_line_y = entry_line_y
        self.exit_line_y = exit_line_y
        self.marker_names = marker_names

        self.num_cycles_seen = 0

        self.is_speedo_above_line = True
        self.is_horn_inside_line = True
        
        self.waiting_for_speedo = False
        self.waiting_for_horn = False

        self.found_inside_horn = False

        self.refresh_state()

    def refresh_state(self) -> None:

        self.state = {
            "cycle_started": False,
            "cycle_ended": False,

            "cycle_start_frame_num": 0,
            "cycle_end_frame_num": 0,

            "marker_frame_nums": {i: (-1, -1) for i in range(len(self.marker_names))}

        }

    def get_frame_nums_of_object(self, obj_id: int):
        return self.state["marker_frame_nums"][obj_id]

    def set_frame_nums_of_object(self, obj_id: int, first_seen: int, last_seen: int):
        self.state["marker_frame_nums"][obj_id] = (first_seen, last_seen)


    def _handle_cycle_start(self, frame_num: int):
        self.state["cycle_started"] = True
        self.state["cycle_start_frame_num"] = frame_num

        # self.state["seen_objects"].append(self.start_marker_object_id)

        print("CYCLE STARTED")
        pprint(self.state)

    def _handle_cycle_end(self, frame_num: int):
        print("CYCLE ENDED")
        self.num_cycles_seen += 1
        
        self.state["cycle_ended"] = True
        self.state["cycle_end_frame_num"] = frame_num

        pprint(self.state)

        for obj_id, (first_seen, last_seen) in self.state["marker_frame_nums"].items():
            if obj_id != self.start_marker_object_id and obj_id != self.end_marker_object_id:
                time_elapsed = get_time_elapsed_ms(first_seen, last_seen, self.stream_fps)
                
                print(f'''
                    {self.marker_names[obj_id]}: 
                    first_seen: {get_time_from_frame(first_seen, self.stream_fps)}, 
                    last_seen: {get_time_from_frame(last_seen, self.stream_fps)}, 
                '''
                )

        self.refresh_state()

    def cycle_started(self):
        return self.state["cycle_started"]

    def get_object_closest_to_line(self, detections, object_id, line_y, check_top=True, buffer=30):
        min_diff = float('inf')
        closest_obj_y  = -1

        for obj_id, x1, y1, x2, y2 in detections:
            if obj_id == object_id:

                y_to_check = y1 if check_top else y2
                diff = abs(y_to_check - line_y)

                if diff < min_diff:
                    min_diff = diff
                    closest_obj_y = y_to_check
        
        if closest_obj_y == -1:
            print(f'{self.marker_names[object_id]} not found...')
            return -1

        if min_diff > buffer:
            print(f'{self.marker_names[object_id]} not close enough to line...')
            return -1
        
        return closest_obj_y 

    def get_object_within_lines(self, detections, object_id, check_top=True):
        for obj_id, x1, y1, x2, y2 in detections:
            if obj_id == object_id:
                y_to_check = y1 if check_top else y2
                if self.entry_line_y < y_to_check < self.exit_line_y:
                    return y_to_check

        return -1

    
    def _process_other_markers(self, frame_num, detections):
        for obj_id, *xyxy in detections:
            if obj_id != self.start_marker_object_id and obj_id != self.end_marker_object_id:
                
                first_seen, _ = self.get_frame_nums_of_object(obj_id)
                
                if first_seen == -1:
                    new_first_seen = frame_num
                    new_last_seen = -1
                else:
                    new_first_seen = first_seen
                    new_last_seen = frame_num  

                self.set_frame_nums_of_object(
                    obj_id, 
                    first_seen=new_first_seen, 
                    last_seen=new_last_seen
                )
                
    # if cycle seen = 0, wait for closest horn to change state
    # horn state changed
    # wait for speedo state change
    # speedo state changed
    # horn must have crossed already, 
    # wait for horn inside lines to change state
    # horn state changed
    # loop to step 3
    def process_detections(
            self,
            detections: Dict[int, Tuple[int, int, int, int]],
            frame_num: int,
            current_frame: npt.NDArray = None,
        ):
        if self.num_cycles_seen == 0:
            print('waiting for first horn to cross...')

            closest_horn_y = self.get_object_closest_to_line(
                            detections=detections,
                            object_id=self.end_marker_object_id,
                            line_y=self.exit_line_y,
                            check_top=False,
                            buffer=100
                        )
            # no horn found, return
            if closest_horn_y == -1:
                return

            # horn crossed
            if closest_horn_y >= self.exit_line_y and self.is_horn_inside_line:
                print("FIRST HORN LEFT")
                print(f'{closest_horn_y = }, {self.exit_line_y = }')
                self.is_horn_inside_line = False

                self._handle_cycle_end(frame_num=frame_num)

                self.waiting_for_speedo = True
                self.waiting_for_horn = False

                return
    

        else:
            # wait for speedo change
            if self.waiting_for_speedo:
                print('waiting for speedo to enter...')

                closest_speedo_y = self.get_object_closest_to_line(
                                    detections=detections,
                                    object_id=self.start_marker_object_id,
                                    line_y=self.entry_line_y,
                                    check_top=True,
                                    buffer=35
                                )
                # no speedo found, return
                if closest_speedo_y == -1:
                    return

                # speedo crossed
                if closest_speedo_y >= self.entry_line_y and self.is_speedo_above_line:
                    print("SPEEDO ENTERED")
                    print(f'{closest_speedo_y = }, {self.entry_line_y = }')
                    self.is_speedo_above_line = False

                    self._handle_cycle_start(frame_num=frame_num)
                    
                    self.waiting_for_speedo = False
                    self.waiting_for_horn = True

                    return

                   

            # at this point, a cycle has started, horn has not left, find other markers
            self._process_other_markers(frame_num=frame_num, detections=detections)


            # find inside horn first
            if not self.found_inside_horn:
                print('waiting to find inside horn...')

                # wait for horn inside lines to change state
                inside_horn_y = self.get_object_within_lines(
                    detections=detections,
                    object_id=self.end_marker_object_id,
                    check_top=False
                )

                if inside_horn_y == -1:
                    return
                
                # we've found a horn inside, set state
                else:
                    print("INSIDE HORN FOUND")
                    print("INSIDE HORN FOUND")
                    print("INSIDE HORN FOUND")
                    print(f'{inside_horn_y = }, {self.entry_line_y = }, {self.exit_line_y = }')
                    self.found_inside_horn = True
                    # set inside_line to True, only then we can observe state change    
                    self.is_horn_inside_line = True
                    return
            
            # found inside horn, wait for state change
            else:
                print('INSIDE HORN FOUND, waiting for inside horn to cross...')

                closest_horn_y = self.get_object_closest_to_line(
                            detections=detections,
                            object_id=self.end_marker_object_id,
                            line_y=self.exit_line_y,
                            check_top=False,
                            buffer=100
                        )


                # no horn found in this frame, return
                if closest_horn_y == -1:
                    return
                
                print('checking for inside horn state change...')

                # horn crossed
                if closest_horn_y >= self.exit_line_y and self.is_horn_inside_line:
                    print("HORN LEFT")
                    print(f'{closest_horn_y = }, {self.exit_line_y = }')
                    self.is_horn_inside_line = False
                    
                    self._handle_cycle_end(frame_num=frame_num)
                    
                    self.waiting_for_speedo = True
                    self.waiting_for_horn = False
                    
                    self.is_speedo_above_line = True

