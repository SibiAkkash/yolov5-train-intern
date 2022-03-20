import torch
import cv2
import time
from PIL import Image
from utils.torch_utils import time_sync

from imutils.video.webcamvideostream import WebcamVideoStream
from vidgear.gears import CamGear
from vidgear.gears import WriteGear

from deep_sort import nn_matching, preprocessing
from deep_sort.track import TrackState
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
import generate_clip_detections as gdet

import clip



def test_on_stream():
    
    model = torch.hub.load('.', 'custom', 'weights/scooter-only-yolov5s.pt', source='local')
    model.conf = 0.6
    model.cuda()
    model.eval()
    
    # cap = WebcamVideoStream(src="/home/sibi/Downloads/cycle_videos/rec_3.mp4").start()

    window_name = "stream"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, 540, 960)

    cap = CamGear(source="/home/sibi/Downloads/cycle_videos/rec_5_cut.mp4")
    
    fps = cap.stream.get(cv2.CAP_PROP_FPS)
    print(f'{fps = }')
    
    cap.start()
    time.sleep(1)
    
    while True:
        img = cap.read()
        
        if img is None:
            break
        
        preds = model(img, size=640)
        preds.display(render=True)
        
        cv2.imshow(window_name, cv2.resize(img, (540, 960)))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.stop()
    # cap.stream.release()


def batch_infer():
    img1 = Image.open('data/images/zidane.jpg')  # PIL image
    img2 = Image.open('data/images/bus.jpg')  # OpenCV image (BGR to RGB)
    imgs = [img1] * 50  # batch of images
        
    with torch.no_grad():
        t1 = time_sync()
        results = model(imgs, size=640)
        t2 = time_sync()

    results.print()
    print(f'\n\n{t2 - t1:3f}s')
    

def track():
    device='cuda:0'
    
    ###### init yolo model ###### 
    model = torch.hub.load('.', 'custom', 'weights/scooter-only-yolov5s.pt', source='local')
    model.conf = 0.6
    model.cuda()
    model.eval()
    
    # ###### init clip model ###### 
    # model_filename = "ViT-B/16"  # all model names in clip/clip.py
    # clip_model, clip_transform = clip.load(name=model_filename, device=device)
    # clip_model.eval()

    # img_encoder = gdet.create_box_encoder(
    #     clip_model, clip_transform, batch_size=1, device=device
    # )
    
    # ###### initialize tracker ###### 
    
    # # params for tracking
    # nms_max_overlap = 1.0
    # max_cosine_distance = 0.4
    # nn_budget = None
    # EXIT_LINE = 600
    
    # metric = nn_matching.NearestNeighborDistanceMetric(
    #     "cosine", max_cosine_distance, nn_budget
    # )
    
    # tracker = Tracker(metric, max_iou_distance=0.7, max_age=50, n_init=10)
    
    # cap = CamGear(source="/home/sibi/Downloads/cycle_videos/rec_5_cut.mp4").start()
    window_name = "stream"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, 540, 960)

    cap = CamGear(source="/home/sibi/Downloads/cycle_videos/rec_3.mp4").start()
    time.sleep(1)
    
    fps = int(cap.stream.get(cv2.CAP_PROP_FPS))
    print(f'{fps = }')
        
    while True:
        img = cap.read()
        
        if img is None:
            break
 
        preds = model(img, size=640)
        preds.display(render=True)
        
        cv2.imshow(window_name, cv2.resize(img, (540, 960)))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        time.sleep(1 / fps)
        
    cv2.destroyAllWindows()
    cap.stop()
    cap.stream.release()
    

if __name__ == "__main__":
    track()