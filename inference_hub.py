
import torch
import cv2
import time
from PIL import Image
from utils.torch_utils import time_sync
from imutils.video.webcamvideostream import WebcamVideoStream
from vidgear.gears import CamGear


model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model.cuda()
model.eval()

def test_on_stream():
    cap = WebcamVideoStream(src="http://localhost:8080/stream/video.mjpeg").start()
    # cap = CamGear(source="http://localhost:8080/stream/video.mjpeg").start()
    time.sleep(2)

    while True:
        img = cap.read()
        
        if img is None:
            break
        
        preds = model(img, size=640)
        preds.display(pprint=True, render=True)
        
        cv2.imshow("stream", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.stop()
    cap.stream.release()


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