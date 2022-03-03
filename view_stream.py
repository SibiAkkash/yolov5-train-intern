import cv2
from utils.datasets import LoadStreams

source = "http://10.131.162.50:8080/stream/video.mjpeg"


def view():
    stream = LoadStreams(sources=source)
    cnt = 0
    for _, _, img0, _, _ in stream:
        # im = cv2.resize(img0[0], (480, 640))
        cnt += 1
        cv2.imshow("stream", img0[0])


def record(save_path: str):
    stream = LoadStreams(sources=source)
    # filename: Input video file
    # fourcc: 4-character code of codec used to compress the frames
    # fps: framerate of videostream
    # framesize: Height and width of frame
    codec = "mp4"
    fps = stream.fps[0]
    # we get flipped video from 
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*codec), fps, (h, w))

    for _, _, img0, _, _ in stream:
        # im = cv2.resize(img0[0], (480, 640))
        cv2.imshow("stream", img0[0])
