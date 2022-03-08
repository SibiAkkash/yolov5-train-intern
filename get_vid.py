import cv2
import sys
import time
import numpy as np
from utils.datasets import LoadStreams
import threading as th
import os


# To view logs  : export OPENCV_LOG_LEVEL=DEBUG; export OPENCVV_VIDEOIO_DEBUG=1
# To run        : python get_vid.py


HOST = '10.131.162.50'
PORT = '8080'
source = f"http://{HOST}:{PORT}/stream/video.mjpeg"

# input resolution: 1920 x 1080
# flipping to 1080 x 1920

codec = "mp4v"
fourcc = cv2.VideoWriter_fourcc(*codec)

def flip_video(orig_path, save_path):
    cap = cv2.VideoCapture(orig_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'{fps = }')

    # we are storing vertical video, so flip resolution
    # width = height, height = width
    vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (h, w))
    print(f'creating flipped video... storing at {save_path}')
    cnt = 0
    while cap.isOpened():
        cnt += 1
        _, img = cap.read()
        # rotate to get vertical image
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        # write to video
        vid_writer.write(img)
        print(cnt)

    print('finished flipping video')
    vid_writer.release()
    cap.release()

def record_stream(duration_min, save_path, flip_path):
    # cap = cv2.VideoCapture(source)
    stream = LoadStreams(sources=source)

    fps = int(stream.fps[0])
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = 1920
    h = 1080

    vid_writer = cv2.VideoWriter(save_path, fourcc, 25, (w, h))

    duration_sec = int(duration_min * 60)

    print(f'recording video for {duration_min} mins, saving at {save_path}')

    frame_count = 0
    start_time = time.time()
    for _, _, img0, _, _ in stream:
        # cv2.imshow("stream", img0[0])
        # print(f'{frame_count = }')
        vid_writer.write(img0[0])
        frame_count += 1

        if time.time() > start_time + duration_sec:
            print('stopping recording')
            vid_writer.release()
            break
        
    print(f'{frame_count = }')
    # actual_num_frames = frame_count
    # ideal_num_frames = fps * duration_sec
    # loss = ideal_num_frames - actual_num_frames
    # loss_percent = 100 * loss / ideal_num_frames
    # print(f'actual num frames:\t{actual_num_frames}')
    # print(f'ideal num frames:\t{ideal_num_frames}', )
    # print(f'intra frame delay led to :\t{loss_percent}% loss')
    flip_video(orig_path=save_path, save_path=flip_path)


def collect_images(save_path):
    os.makedirs(save_path, exist_ok = True)
    stream = LoadStreams(sources=source)
    cnt = 0
    for _, _, img0, _, _ in stream:
        cv2.imshow("s", cv2.resize(img0[0], (540, 960)))

        key = cv2.waitKey(5)
        if key == ord('c'):
            cnt += 1
            cv2.imwrite(f"{save_path}/{cnt:04d}.jpg", img0[0])
            print(cnt, ' captured image', img0[0].shape)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


collect_images(save_path="images_2")

# record_stream(duration_min=0.1, save_path="videos/test_2.mp4", flip_path="videos/test_2_flip.mp4")
# flip_video("videos/test_2.mp4", "videos/test_2_flip.mp4")

