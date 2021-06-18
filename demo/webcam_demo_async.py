import argparse
import time
from collections import deque
from queue import Queue
from threading import Lock, Thread

import cv2
import numpy as np

from mmpose.apis import init_pose_model

# from mmpose.apis import (get_track_id, inference_top_down_pose_model,
#                          init_pose_model, vis_pose_result)
# from mmpose.core import apply_bugeye_effect, apply_sunglasses_effect
# from mmpose.utils import StopWatch

try:
    from mmdet.apis import inference_detector, init_detector  # noqa: F401
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument(
        '--det_config',
        type=str,
        default='demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py',
        help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmdetection/v2.0/yolo/'
        'yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--enable_human_pose',
        type=int,
        default=1,
        help='Enable human pose estimation')
    parser.add_argument(
        '--enable_animal_pose',
        type=int,
        default=1,
        help='Enable animal pose estimation')
    parser.add_argument(
        '--human_pose_config',
        type=str,
        default='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/'
        'hrnet_w48_coco_256x192.py',
        help='Config file for human pose')
    parser.add_argument(
        '--human_pose_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/'
        'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
        help='Checkpoint file for human pose')
    parser.add_argument(
        '--human_det_ids',
        type=int,
        default=[1],
        nargs='+',
        help='Object category label of human in detection results.'
        'Default is [1(person)], following COCO definition.')
    parser.add_argument(
        '--animal_pose_config',
        type=str,
        default='configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'animalpose/hrnet_w32_animalpose_256x256.py',
        help='Config file for animal pose')
    parser.add_argument(
        '--animal_pose_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/animal/hrnet/'
        'hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
        help='Checkpoint file for animal pose')
    parser.add_argument(
        '--animal_det_ids',
        type=int,
        default=[16, 17, 18, 19, 20],
        nargs='+',
        help='Object category label of animals in detection results'
        'Default is [16(cat), 17(dog), 18(horse), 19(sheep), 20(cow)], '
        'following COCO definition.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.5,
        help='bbox score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--sunglasses', action='store_true', help='Apply `sunglasses` effect.')
    parser.add_argument(
        '--bugeye', action='store_true', help='Apply `bug-eye` effect.')

    parser.add_argument(
        '--out-video-file',
        type=str,
        default=None,
        help='Record the video into a file. This may reduce the frame rate')

    parser.add_argument(
        '--buffer-size',
        type=int,
        default=1,
        help='Frame buffer size. Default is 1')

    return parser.parse_args()


def process_mmdet_results(mmdet_results, class_names=None, cat_ids=1):
    """Process mmdet results to mmpose input format.

    Args:
        mmdet_results: raw output of mmdet model
        class_names: class names of mmdet model
        cat_ids (int or List[int]): category id list that will be preserved
    Returns:
        List[Dict]: detection results for mmpose input
    """
    if isinstance(mmdet_results, tuple):
        mmdet_results = mmdet_results[0]

    if not isinstance(cat_ids, (list, tuple)):
        cat_ids = [cat_ids]

    # only keep bboxes of interested classes
    bbox_results = [mmdet_results[i - 1] for i in cat_ids]
    bboxes = np.vstack(bbox_results)

    # get textual labels of classes
    labels = np.concatenate([
        np.full(bbox.shape[0], i - 1, dtype=np.int32)
        for i, bbox in zip(cat_ids, bbox_results)
    ])
    if class_names is None:
        labels = [f'class: {i}' for i in labels]
    else:
        labels = [class_names[i] for i in labels]

    det_results = []
    for bbox, label in zip(bboxes, labels):
        det_result = dict(bbox=bbox, label=label)
        det_results.append(det_result)
    return det_results


def read_camera(args):

    # init video reader
    vid_cap = cv2.VideoCapture(args.cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={args.cam_id})')
        exit()

    while True:
        # capture a camera frame
        ret_val, frame = vid_cap.read()
        if not ret_val:
            break
        timestamp = time.time()
        frame_buffer.put((timestamp, ret_val))

    vid_cap.release()


def inference_detection(args):
    print('Thread "det" started')
    time.sleep(2)


def inference_pose(args):
    print('Thread "pose" started')
    time.sleep(2)


def main():
    global frame_buffer, det_result_queue, result_queue, det_model, \
        pose_model_list, pose_history_list, det_result_queue_mutex,\
        result_queue_mutex

    args = parse_args()

    assert has_mmdet, 'Please install mmdet to run the demo.'
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # build detection model
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build pose models
    pose_model_list = []
    if args.enable_human_pose:
        pose_model = init_pose_model(
            args.human_pose_config,
            args.human_pose_checkpoint,
            device=args.device.lower())
        model_info = {
            'name': 'HumanPose',
            'model': pose_model,
            'cat_ids': args.human_det_ids,
            'bbox_color': (148, 139, 255),
        }
        pose_model_list.append(model_info)
    if args.enable_animal_pose:
        pose_model = init_pose_model(
            args.animal_pose_config,
            args.animal_pose_checkpoint,
            device=args.device.lower())
        model_info = {
            'name': 'AnimalPose',
            'model': pose_model,
            'cat_ids': args.animal_det_ids,
            'bbox_color': 'cyan',
        }
        pose_model_list.append(model_info)

    # store pose history for pose tracking
    pose_history_list = []
    for _ in range(len(pose_model_list)):
        pose_history_list.append({'pose_results_last': [], 'next_id': 0})

    # queue of input frames
    # (timestamp, frame)
    frame_buffer = Queue(maxsize=args.buffer_size)

    # queue of detection results
    # (timestamp, frame, messages, det_result)
    det_result_queue = deque(maxlen=1)
    det_result_queue_mutex = Lock()

    # queue of detection/pose results
    # (timestamp, frame, messages, det_result, pose_result_list)
    result_queue = deque(maxlen=1)
    result_queue_mutex = Lock()

    try:
        t_input = Thread(target=read_camera, args=(args, ), daemon=True)
        t_det = Thread(target=inference_detection, args=(args, ), daemon=True)
        t_pose = Thread(target=inference_pose, args=(args, ), daemon=True)

        t_input.start()
        t_det.start()
        t_pose.start()
        t_input.join()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
