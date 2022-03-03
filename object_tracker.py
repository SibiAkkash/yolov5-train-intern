import argparse
from collections import defaultdict
import csv
from itertools import cycle
from multiprocessing.spawn import prepare
import os
import sys
from pathlib import Path

import cv2
import deep_sort
from deep_sort.track import TrackState
from helpers import draw_text_with_box
import torch
import torch.backends.cudnn as cudnn

from visual_inspector import VisualInspector
import numpy as np

import clip


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from deep_sort import nn_matching, preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
import generate_clip_detections as gdet


@torch.no_grad()
def run(
    weights=ROOT / "yolov5s.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob, 0 for webcam
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference,
    show_overlay=False,  # show debug overlay,
    cycle_times_save_path=ROOT
    / "cycle_times/cycle_times.txt",  # file to save cycle times
):

    # params for tracking
    nms_max_overlap = 1.0
    max_cosine_distance = 0.4
    nn_budget = None

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = (
        model.stride,
        model.names,
        model.pt,
        model.jit,
        model.onnx,
        model.engine,
    )
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # print(model.names)
    # ['horn', 'speedo', 'exposed_fork', 'torque_tool_hanging', 'torque_tool_inserted', 'ball_bearing_tool', 'QR_code_scanner', 'wheel_with_fender']

    # Half
    half &= (
        pt or jit or onnx or engine
    ) and device.type != "cpu"  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # init clip model
    model_filename = "ViT-B/16"  # all model names in clip/clip.py
    clip_model, clip_transform = clip.load(
        name=model_filename, device=device
    )
    clip_model.eval()
    
    img_encoder = gdet.create_box_encoder(
        clip_model, clip_transform, batch_size=1, device=device
    )

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )

    # initialize tracker
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=100, n_init=10)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        fps = dataset.fps[0]
        bs = len(dataset)  # batch_size
        stream = True
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
        fps = dataset.cap.get(cv2.CAP_PROP_FPS)
        stream = False

    print(f"{fps = }")

    vid_path, vid_writer = [None] * bs, [None] * bs
    video_writer = None
    writers = defaultdict(None)
    bbox_sizes = defaultdict(list)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0, 0.0], 0

    for path, im, im0s, vid_cap, s in dataset:
        
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = (
            increment_path(save_dir / Path(path).stem, mkdir=True)
            if visualize
            else False
        )
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            bboxes = []
            confs = []
            class_nums = []

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # bbox in form
                trans_bboxes = det[:, :4].clone()
                trans_bboxes[:, 2:] -= trans_bboxes[
                    :, :2
                ]  # or, trans_bboxes = xyxy2xywh(det[:, :4])

                bboxes = trans_bboxes[:, :4].cpu()  # why indexing again ?
                confs = det[:, 4].cpu()
                class_nums = det[:, -1].cpu()

            # encode yolo detections and feed to tracker
            t4 = time_sync()
            features = img_encoder(im0, bboxes)
            t5 = time_sync()
            dt[3] += t5 - t4

            detections = [
                Detection(bbox, conf, class_num, feature)
                for bbox, conf, class_num, feature in zip(
                    bboxes, confs, class_nums, features
                )
            ]

            # run non-maxima suppression
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            class_nums = np.array([d.class_num for d in detections])

            # supress overlapping detections
            indices = preprocessing.non_max_suppression(
                boxes, class_nums, nms_max_overlap, scores
            )

            detections = [detections[i] for i in indices]

            # call the tracker
            tracker.predict()
            tracker.update(detections)

            if len(tracker.tracks):
                print("[Tracks]", len(tracker.tracks))

            H, W, _ = im0.shape
            EXIT_LINE = 700

            for track in tracker.tracks:
                xyxy = track.to_tlbr()
                bbox = xyxy
                class_num = track.class_num
                class_name = names[int(class_num)]

                if not track.is_confirmed() or track.time_since_update > 1:
                    print(
                        f"NOT CONFIRMED\tTracker ID: {str(track.track_id)}, Class: {class_name}"
                    )
                    continue

                cx = (int(bbox[0]) + int(bbox[2])) // 2
                cy = (int(bbox[1]) + int(bbox[3])) // 2
                if cy >= 700:
                    track.state = TrackState.Deleted
                    print(f"Track {track.track_id} DELETED DELETED DELETED !!!!")
                    continue

                # BBox Coords (xmin, ymin, xmax, ymax): {(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))}
                print(f"Tracker ID: {str(track.track_id)}, Class: {class_name}")

                if save_img or view_img:
                    x1, y1, x2, y2 = bbox
                    x1 = int(min(max(x1, 0), W))
                    x2 = int(min(max(x2, 0), W))
                    y1 = int(min(max(y1, 0), H))
                    y2 = int(min(max(y2, 0), H))
                    
                    label = f"{class_name} #{track.track_id}"
                    
                    annotator.box_label(
                        bbox,
                        label,
                        color=colors(int(class_num), True),
                        show_center=True,
                    )

                    # plot prev bbox centers
                    for cx, cy in track.prev_locs:
                        cv2.circle(
                            im0,
                            (int(cx), int(cy)),
                            2,
                            # color=colors(int(class_num), True),
                            color=(0, 173, 255),
                            thickness=-1,
                        )
                    
                                        
                    # draw_text_with_box(im0, text=f"{x2 - x1} x {y2- y1}", base_coords=(cx - 30, cy-20), fontScale=0.5, padding=2, thickness=1)

                    # crop = save_one_box(
                    #     xyxy, im0, save=True, file=Path("./crop_images/img")
                    # )
                    
                bbox_sizes[track.track_id].append((x2 - x1, y2 - y1))
                    
                # if track.track_id not in writers:
                #     print(f"creating vid writer for scooter {track.track_id}")
                #     writers[track.track_id] = cv2.VideoWriter(
                #         f"crop_videos/scooter_{track.track_id}.mp4",
                #         cv2.VideoWriter_fourcc(*"mp4v"),
                #         fps,
                #         (W, H),
                #         True
                #     )

            # Stream results
            im0 = annotator.result()
            cv2.line(im0, (0, EXIT_LINE), (W, EXIT_LINE), (0, 255, 0), 2)

            if view_img:
                cv2.imshow(str(p), im0)

                if cv2.waitKey(1) == ord("q"):
                    print("trying to quit")
                    print("releasing vid writer")
                    vid_cap.release()
                    # video_writer.release()
                    raise StopIteration
                    # break
            # video_writer.write(im0)

    print("releasing vid writer")
    # video_writer.release()
    
    # write bbox sizes to csv file
    with open("cycle_times/bbox_sizes.csv", 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['scooter_id', 'w', 'h'])
        for scooter_id in bbox_sizes:
            for (w, h) in bbox_sizes[scooter_id]:
                csv_writer.writerow([scooter_id, w, h])
    

    # Print results
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
        % t
    )
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/coco128.yaml",
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument(
        "--show-overlay", action="store_true", help="show debug overlay"
    )
    parser.add_argument(
        "--cycle-times-save-path",
        type=str,
        default=ROOT / "cycle_times/cycle_times.txt",
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
