import cv2
from utils.datasets import LoadStreams

source = "http://10.131.162.50:8080/stream/video.mjpeg"


def view():
    stream = LoadStreams(sources=source)
    # cv2.namedWindow("stream", cv2.WINDOW_NORMAL)
    for _, _, img0, _, _ in stream:
        cv2.imshow("stream", cv2.resize(img0[0], (360, 640)))


if __name__ == "__main__":
    view()