import cv2
from utils.datasets import LoadStreams

source = 'http://10.131.162.50:8080/stream/video.mjpeg'

stream = LoadStreams(sources=source)

for _, _, img0, _, _ in stream:
    im = cv2.resize(img0[0], (540, 960))
    cv2.imshow("stream", im)

