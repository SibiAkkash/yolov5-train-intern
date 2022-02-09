from __future__ import annotations
from operator import is_

from typing import Dict, List, Tuple
from pprint import pprint
import numpy as np
import numpy.typing as npt
from pathlib import Path
import time
import copy
from dataclasses import dataclass, field
from helpers import get_time_elapsed_ms, get_time_from_frame, plot_last_2_cycles


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
        start_marker_check_top: bool,
        end_marker_check_top: bool,
        stream_fps: float,
        entry_line_y: int,
        exit_line_y: int,
        marker_names: List[str],
        cycle_times_save_path: str = None
    ):
        self.start_marker_object_id = start_marker_object_id
        self.end_marker_object_id = end_marker_object_id
        self.stream_fps = stream_fps
        self.entry_line_y = entry_line_y
        self.exit_line_y = exit_line_y
        self.marker_names = marker_names

        self.start_marker_check_top = start_marker_check_top
        self.end_marker_check_top = end_marker_check_top

        self.cycle_times_save_path = cycle_times_save_path

        self.num_cycles_seen = 0

        self.is_speedo_above_line = True
        self.is_horn_inside_line = True
        
        # when we initialise the inspector, we wait for one horn to cross, then start from the next cycle
        self.waiting_for_horn_to_cross = True
        self.waiting_for_speedo_to_cross = False
        self.waiting_to_see_inside_horn = False

        self.status = "waiting for end_marker cross"

        self.found_inside_horn = False

        self.cycle_cache = []

        self.refresh_state()

    def refresh_state(self) -> None:
        self.state = {
            "cycle_num": -1,

            "cycle_started": False,
            "cycle_ended": False,

            "cycle_start_frame_num": 0,
            "cycle_start_frame_time": 0,
            
            "cycle_start_sys_time": 0,
            "cycle_end_sys_time": 0,
            
            "cycle_end_frame_num": 0,
            "cycle_end_frame_time": 0,

            "marker_frame_nums": {i: (-1, -1) for i in range(len(self.marker_names))},
            "marker_frame_times": {i: (-1, -1) for i in range(len(self.marker_names))},
            "marker_times_relative": {i: (-1, -1) for i in range(len(self.marker_names))},
            "marker_time_elapsed": [0 for i in range(len(self.marker_names))]
        }

    def get_frame_nums_of_object(self, obj_id: int):
        return self.state["marker_frame_nums"][obj_id]

    def get_obj_frame_times_relative(self, obj_id: int):
        return self.state["marker_times_relative"][obj_id]

    def set_obj_frame_times_relative(self, obj_id: int, first_seen_t, last_seen_t):
        self.state["marker_times_relative"][obj_id] = (first_seen_t, last_seen_t)

    def set_frame_nums_of_object(self, obj_id: int, first_seen: int, last_seen: int):
        first_seen_time = get_time_from_frame(first_seen, self.stream_fps)
        last_seen_time = get_time_from_frame(last_seen, self.stream_fps)
        self.state["marker_frame_nums"][obj_id] = (first_seen, last_seen)
        self.state["marker_frame_times"][obj_id] = (first_seen_time, last_seen_time)
        
    def _handle_cycle_start(self, frame_num: int):
        print("CYCLE STARTED")
        self.state["cycle_started"] = True

        self.state["cycle_start_frame_num"] = frame_num
        self.state["cycle_start_frame_time"] = get_time_from_frame(frame_num, self.stream_fps)
        
        self.state["cycle_start_sys_time"] = time.time()

        pprint(self.state)

    def _handle_cycle_end(self, frame_num: int):
        print("CYCLE ENDED")
        self.num_cycles_seen += 1
        self.state["cycle_num"] = self.num_cycles_seen
        
        self.state["cycle_ended"] = True
        
        self.state["cycle_end_frame_num"] = frame_num
        self.state["cycle_end_frame_time"] = get_time_from_frame(frame_num, self.stream_fps)
        
        self.state["cycle_end_sys_time"] = time.time()


        for obj_id, (first_seen, last_seen) in self.state["marker_frame_nums"].items():
            time_elapsed = get_time_elapsed_ms(first_seen, last_seen, self.stream_fps)
            self.state["marker_time_elapsed"][obj_id] = time_elapsed
            
            # print(f'''
            #     {self.marker_names[obj_id]}: 
            #     first_seen: {get_time_from_frame(first_seen, self.stream_fps)}, 
            #     last_seen: {get_time_from_frame(last_seen, self.stream_fps)}, 
            # '''
            # )

        # cache to plot
        # ignore first cycle
        if self.num_cycles_seen > 1:
            self.cycle_cache.append(copy.deepcopy(self.state))
            # write cycle time to file
            # with open(self.cycle_times_save_path, 'a') as f:
            #     global_cycle_time = self.state["cycle_end_sys_time"] - self.state["cycle_start_sys_time"]
            #     global_cycle_time = str(round(global_cycle_time, 2))
            #     f.write(f"{global_cycle_time}\n")

        # store only 2 cycles
        if len(self.cycle_cache) > 2:
            self.cycle_cache.pop(0)

        pprint(self.state)

        self.refresh_state()

    def cycle_started(self):
        return self.state["cycle_started"]

    def plot_cycles(self, fig, ax):
        return plot_last_2_cycles(fig, ax, self.cycle_cache)

    def get_object_closest_to_line(self, detections, object_id, line_y, check_top=True, buffer=30, min_bbox_height=30):
        min_diff = float('inf')
        closest_obj_y  = -1

        for obj_id, x1, y1, x2, y2 in detections:
            if obj_id == object_id:
                # check crossing with bbox top or bbox bottom
                y_to_check = y1 if check_top else y2
                diff = abs(y_to_check - line_y)
                
                # check size of bbox, only for horn
                is_speedo = object_id == self.start_marker_object_id
                too_small = False if is_speedo else ((y2 - y1) < min_bbox_height)

                if not is_speedo:
                    print('bbox height: ', y2 - y1)

                if diff < min_diff and not too_small:
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
                # if self.entry_line_y < y_to_check < self.exit_line_y:
                if self.entry_line_y < y_to_check < self.exit_line_y - 100:
                    return y_to_check

        print(f'{self.marker_names[object_id]} not found within lines...')
        return -1

    def _process_other_markers(self, frame_num, detections):
        for obj_id, *xyxy in detections:                
            # using frame numbers
            first_seen, _ = self.get_frame_nums_of_object(obj_id)
            if first_seen == -1:
                new_first_seen = frame_num
                new_last_seen = -1
            else:
                new_first_seen = first_seen
                new_last_seen = frame_num  

            self.set_frame_nums_of_object(obj_id, new_first_seen, new_last_seen)

            # using sys time
            first_seen_t, _ = self.get_obj_frame_times_relative(obj_id)
    
            if first_seen_t == -1:
                new_fst_t = time.time() - self.state["cycle_start_sys_time"]
                new_lst_t = -1
            else:
                new_fst_t = first_seen_t
                new_lst_t = time.time() - self.state["cycle_start_sys_time"]

            self.set_obj_frame_times_relative(obj_id, new_fst_t, new_lst_t)
                
    def _wait_for_end_marker_cross(
            self,
            detections: Dict[int, Tuple[int, int, int, int]],
            frame_num: int
        ):
        # dont capture markers until first horn has crossed
        if self.num_cycles_seen > 0:
            self._process_other_markers(frame_num=frame_num, detections=detections)
            
        print('waiting for horn to cross...', end='\t')

        closest_horn_y = self.get_object_closest_to_line(
                        detections=detections,
                        object_id=self.end_marker_object_id,
                        line_y=self.exit_line_y,
                        check_top=self.end_marker_check_top,
                        buffer=30
                    )

        # no horn found in this frame, return
        if closest_horn_y == -1:
            return

        print('checking for horn state change...')

        # horn crossed
        if closest_horn_y >= self.exit_line_y and self.is_horn_inside_line:
            print("HORN LEFT\nHORN LEFT\nHORN LEFT\n")
            print(f'{closest_horn_y = }, {self.exit_line_y = }')
            self.is_horn_inside_line = False

            self._handle_cycle_end(frame_num=frame_num)

            # next state
            self.waiting_for_speedo_to_cross = True
            # set inside_line to True, only then we can observe state change    
            # assuming a speedo is about to enter
            self.is_speedo_above_line = True
            
            self.waiting_for_horn_to_cross = False
            self.waiting_to_see_inside_horn = False

            self.status = "waiting for start_marker cross"


    def _wait_for_start_marker_cross(
        self,
        detections: Dict[int, Tuple[int, int, int, int]],
        frame_num: int
    ):
        print('waiting for speedo to enter...', end='\t')

        closest_speedo_y = self.get_object_closest_to_line(
                            detections=detections,
                            object_id=self.start_marker_object_id,
                            line_y=self.entry_line_y,
                            check_top=self.cycle_started,
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
            
            # next state
            self.waiting_to_see_inside_horn = True
            self.waiting_for_speedo_to_cross = False
            self.waiting_for_horn_to_cross = False

            self.status = "waiting for inside end_marker"



    def _wait_for_end_marker_inside(
        self,
        detections: Dict[int, Tuple[int, int, int, int]],
        frame_num: int
    ):
        # at this point, a cycle has started, horn has not left, find other markers
        self._process_other_markers(frame_num=frame_num, detections=detections)

        # wait for horn inside lines to change state
        print('waiting to find inside horn...', end='\t')

        inside_horn_y = self.get_object_within_lines(
            detections=detections,
            object_id=self.end_marker_object_id,
            check_top=self.end_marker_check_top
        )

        # return if we dont find horn inside lines
        if inside_horn_y == -1:
            return

        # we've found a horn inside, set state
        print("INSIDE HORN FOUND\nINSIDE HORN FOUND\nINSIDE HORN FOUND")
        print(f'{inside_horn_y = }, {self.entry_line_y = }, {self.exit_line_y = }')
        
        # next state
        self.waiting_for_horn_to_cross = True
        # set inside_line to True, only then we can observe state change    
        self.is_horn_inside_line = True
        
        self.waiting_for_speedo_to_cross = False
        self.waiting_to_see_inside_horn = False

        self.status = "waiting for end_marker cross"


    # 1. if cycle seen = 0, wait for closest end_marker (horn) to change state
    # 2. end_marker (horn) state changed
    # 3. wait for start_marker (speedo) state change
    # 4. start_marker (speedo) state changed
    # 5. end_marker (horn) must have crossed already, 
    # 6. wait for end_marker (horn) inside lines to change state
    # 7. end_marker (horn) state changed
    # 8. loop to step 3
    def process_detections_2(
        self,
        detections: Dict[int, Tuple[int, int, int, int]],
        frame_num: int,
        current_frame: npt.NDArray = None,
    ):
        print(f'processing frame {frame_num}')
        if self.waiting_for_horn_to_cross:
            return self._wait_for_end_marker_cross(detections=detections, frame_num=frame_num)
            
        if self.waiting_for_speedo_to_cross:
            return self._wait_for_start_marker_cross(detections=detections, frame_num=frame_num)
            
        if self.waiting_to_see_inside_horn:
            return self._wait_for_end_marker_inside(detections=detections, frame_num=frame_num)
                

