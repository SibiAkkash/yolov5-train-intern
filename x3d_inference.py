import sys
import os
from pathlib import Path
import time
from slowfast.config.defaults import get_cfg
from pprint import pprint

# FILE = Path(__file__).resolve()
# print(FILE)
# ROOT = FILE.parents[0]  # root directory
# print(ROOT)
# if str(ROOT) not in sys.path:
#     print('root not in syspath')
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# print(ROOT)

from slowfast_tools.demo_net import demo
import multiprocessing as mp

DIR_ROOT = "/media/sibi/DATA/dev/ai/internship"

class ActionPredictionManager:
    class _PredWorker(mp.Process):
        def __init__(self, video_queue, cfg):
            self.video_queue = video_queue
            self.cfg = cfg
            super().__init__()

        def run(self):
            while True:
                input_video_path = self.video_queue.get()
                if isinstance(input_video_path, _StopToken):
                    break

                video_name = input_video_path.split("/")[-1]
                output_video_path = os.path.join(
                    DIR_ROOT, "slowfast/output_videos", "from_yolo.mp4"
                )
                # print(input_video_path)
                # print(output_video_path)

                self.cfg.DEMO.INPUT_VIDEO = input_video_path
                self.cfg.DEMO.OUTPUT_FILE = output_video_path

                demo(self.cfg)
                print(f'\n\n\n done processing {input_video_path} \n\n\n')

    def __init__(self, num_workers=1):
        self.load_config()

        self.video_queue = mp.Queue()
        self.procs = []

        for _ in range(num_workers):
            self.procs.append(
                ActionPredictionManager._PredWorker(self.video_queue, self.cfg)
            )

        for p in self.procs:
            p.start()

    def load_config(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file("./slowfast_configs/Kinetics/pytorchvideo/X3D_M.yaml")
        self.cfg.DEMO.BUFFER_SIZE = 0
        self.cfg.DEMO.NUM_VIS_INSTANCES = 1

    def put(self, path):
        print('incoming video:', path)
        self.video_queue.put(path)

    def shutdown(self):
        for _ in self.procs:
            self.video_queue.put(_StopToken())


class _StopToken:
    pass


def another_task():
    ac = ActionPredictionManager(num_workers=1)
    
    for i in range(1, 101):
        print(i)
        if i % 50 == 0:
            ac.put('./videos/crops3/scooter_1.mp4')

        time.sleep(0.5)            

    ac.shutdown()

if __name__ == "__main__":
    another_task()
    # pass
    