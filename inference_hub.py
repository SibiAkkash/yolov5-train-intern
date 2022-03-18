
import torch
import cv2
import time
from PIL import Image
from utils.torch_utils import time_sync
from imutils.video.webcamvideostream import WebcamVideoStream
from vidgear.gears import CamGear


model = torch.hub.load('.', 'custom', 'weights/scooter-only-yolov5s.pt', source='local')
model.conf = 0.6
model.cuda()
model.eval()

def test_on_stream():
    # cap = WebcamVideoStream(src="/home/sibi/Downloads/cycle_videos/rec_3.mp4").start()
    cap = CamGear(source="/home/sibi/Downloads/cycle_videos/rec_5_cut.mp4").start()
    time.sleep(2)

    window_name = "stream"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, 540, 960)

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
    

if __name__ == "__main__":
    test_on_stream()