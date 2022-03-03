import cv2
from utils.datasets import LoadStreams

source = 'http://10.131.162.50:8080/stream/video.mjpeg'


def view():
    stream = LoadStreams(sources=source)
    cnt = 0
    for _, _, img0, _, _ in stream:
        # im = cv2.resize(img0[0], (480, 640))
        cnt += 1
        cv2.imshow("stream", img0[0])

    print(cnt)

view()