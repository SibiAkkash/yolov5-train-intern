# import cv2
# import torch
# import os

# model = torch.hub.load('ultralytics/yolov5', 'yolov5x', force_reload=True)
# model.cuda()
# model.eval()

# img = cv2.imread("data/images/bus.jpg")

# results = model(img)

# results.print()
# results.show()

from utils.torch_utils import time_sync

import torch
import cv2
import time
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model.cuda()
# model.eval()

# cap = cv2.VideoCapture("http://localhost:8080/stream/video.mjpeg")

# time.sleep(2)

# while cap.isOpened():
#     ret, img = cap.read()
    
#     if not ret:
#         break
    
#     preds = model(img, size=640)

#     # preds.display()
#     preds.display(pprint=True, render=True)
    
#     cv2.imshow("stream", img)
    
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()

img1 = Image.open('data/images/zidane.jpg')  # PIL image
img2 = Image.open('data/images/bus.jpg')  # OpenCV image (BGR to RGB)
imgs = [img1] * 50  # batch of images
    
with torch.no_grad():
    t1 = time_sync()
    results = model(imgs, size=640)
    t2 = time_sync()
    

results.print()

print(f'\n\n{t2 - t1:3f}s')