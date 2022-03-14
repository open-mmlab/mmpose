# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from queue import Queue
from threading import Event, Lock, Thread

import cv2
import numpy as np

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, vis_pose_result)
from mmpose.core import apply_bugeye_effect, apply_sunglasses_effect
from mmpose.utils import StopWatch

try:
    from mmdet.apis import inference_detector, init_detector
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
    parser.add_argument('--cam-id', type=str, default='0')
    parser.add_argument(
        '--det-config',
        type=str,
        default='demo/mmdetection_cfg/'
        'ssdlite_mobilenetv2_scratch_600e_coco.py',
        help='Config file for detection')
    parser.add_argument(
        '--det-checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmdetection/v2.0/ssd/'
        'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
        'scratch_600e_coco_20210629_110627-974d9307.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--enable-human-pose',
        type=int,
        default=1,
        help='Enable human pose estimation')
    parser.add_argument(
        '--enable-animal-pose',
        type=int,
        default=0,
        help='Enable animal pose estimation')
    parser.add_argument(
        '--human-pose-config',
        type=str,
        default='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'coco-wholebody/vipnas_res50_coco_wholebody_256x192_dark.py',
        help='Config file for human pose')
    parser.add_argument(
        '--human-pose-checkpoint',
        type=str,
        default='https://download.openmmlab.com/'
        'mmpose/top_down/vipnas/'
        'vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth',
        help='Checkpoint file for human pose')
    parser.add_argument(
        '--human-det-ids',
        type=int,
        default=[1],
        nargs='+',
        help='Object category label of human in detection results.'
        'Default is [1(person)], following COCO definition.')
    parser.add_argument(
        '--animal-pose-config',
        type=str,
        default='configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'animalpose/hrnet_w32_animalpose_256x256.py',
        help='Config file for animal pose')
    parser.add_argument(
        '--animal-pose-checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/animal/hrnet/'
        'hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
        help='Checkpoint file for animal pose')
    parser.add_argument(
        '--animal-det-ids',
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
        '--vis-mode',
        type=int,
        default=2,
        help='0-none. 1-detection only. 2-detection and pose.')
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
        '--out-video-fps',
        type=int,
        default=20,
        help='Set the FPS of the output video file.')

    parser.add_argument(
        '--buffer-size',
        type=int,
        default=-1,
        help='Frame buffer size. If set -1, the buffer size will be '
        'automatically inferred from the display delay time. Default: -1')

    parser.add_argument(
        '--inference-fps',
        type=int,
        default=10,
        help='Maximum inference FPS. This is to limit the resource consuming '
        'especially when the detection and pose model are lightweight and '
        'very fast. Default: 10.')

    parser.add_argument(
        '--display-delay',
        type=int,
        default=0,
        help='Delay the output video in milliseconds. This can be used to '
        'align the output video and inference results. The delay can be '
        'disabled by setting a non-positive delay time. Default: 0')

    parser.add_argument(
        '--synchronous-mode',
        action='store_true',
        help='Enable synchronous mode that video I/O and inference will be '
        'temporally aligned. Note that this will reduce the display FPS.')

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

    if isinstance(class_names, str):
        class_names = (class_names, )

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


def read_camera():
    # init video reader
    print('Thread "input" started')
    cam_id = args.cam_id
    if cam_id.isdigit():
        cam_id = int(cam_id)
    vid_cap = cv2.VideoCapture(cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={cam_id})')
        exit()

    while not event_exit.is_set():
        # capture a camera frame
        ret_val, frame = vid_cap.read()
        if ret_val:
            ts_input = time.time()

            event_inference_done.clear()
            with input_queue_mutex:
                input_queue.append((ts_input, frame))

            if args.synchronous_mode:
                event_inference_done.wait()

            frame_buffer.put((ts_input, frame))
        else:
            # input ending signal
            frame_buffer.put((None, None))
            break

    vid_cap.release()


def inference_detection():
    print('Thread "det" started')
    stop_watch = StopWatch(window=10)
    min_interval = 1.0 / args.inference_fps
    _ts_last = None  # timestamp when last inference was done

    while True:
        while len(input_queue) < 1:
            time.sleep(0.001)
        with input_queue_mutex:
            ts_input, frame = input_queue.popleft()
        # inference detection
        with stop_watch.timeit('Det'):
            mmdet_results = inference_detector(det_model, frame)

        t_info = stop_watch.report_strings()
        with det_result_queue_mutex:
            det_result_queue.append((ts_input, frame, t_info, mmdet_results))

        # limit the inference FPS
        _ts = time.time()
        if _ts_last is not None and _ts - _ts_last < min_interval:
            time.sleep(min_interval - _ts + _ts_last)
        _ts_last = time.time()


def inference_pose():
    print('Thread "pose" started')
    stop_watch = StopWatch(window=10)

    while True:
        while len(det_result_queue) < 1:
            time.sleep(0.001)
        with det_result_queue_mutex:
            ts_input, frame, t_info, mmdet_results = det_result_queue.popleft()

        pose_results_list = []
        for model_info, pose_history in zip(pose_model_list,
                                            pose_history_list):
            model_name = model_info['name']
            pose_model = model_info['model']
            cat_ids = model_info['cat_ids']
            pose_results_last = pose_history['pose_results_last']
            next_id = pose_history['next_id']

            with stop_watch.timeit(model_name):
                # process mmdet results
                det_results = process_mmdet_results(
                    mmdet_results,
                    class_names=det_model.CLASSES,
                    cat_ids=cat_ids)

                # inference pose model
                dataset_name = pose_model.cfg.data['test']['type']
                pose_results, _ = inference_top_down_pose_model(
                    pose_model,
                    frame,
                    det_results,
                    bbox_thr=args.det_score_thr,
                    format='xyxy',
                    dataset=dataset_name)

                pose_results, next_id = get_track_id(
                    pose_results,
                    pose_results_last,
                    next_id,
                    use_oks=False,
                    tracking_thr=0.3,
                    use_one_euro=True,
                    fps=None)

                pose_results_list.append(pose_results)

                # update pose history
                pose_history['pose_results_last'] = pose_results
                pose_history['next_id'] = next_id

        t_info += stop_watch.report_strings()
        with pose_result_queue_mutex:
            pose_result_queue.append((ts_input, t_info, pose_results_list))

        event_inference_done.set()


def display():
    print('Thread "display" started')
    stop_watch = StopWatch(window=10)

    # initialize result status
    ts_inference = None  # timestamp of the latest inference result
    fps_inference = 0.  # infenrece FPS
    t_delay_inference = 0.  # inference result time delay
    pose_results_list = None  # latest inference result
    t_info = []  # upstream time information (list[str])

    # initialize visualization and output
    sunglasses_img = None  # resource image for sunglasses effect
    text_color = (228, 183, 61)  # text color to show time/system information
    vid_out = None  # video writer

    # show instructions
    print('Keyboard shortcuts: ')
    print('"v": Toggle the visualization of bounding boxes and poses.')
    print('"s": Toggle the sunglasses effect.')
    print('"b": Toggle the bug-eye effect.')
    print('"Q", "q" or Esc: Exit.')

    while True:
        with stop_watch.timeit('_FPS_'):
            # acquire a frame from buffer
            ts_input, frame = frame_buffer.get()
            # input ending signal
            if ts_input is None:
                break

            img = frame

            # get pose estimation results
            if len(pose_result_queue) > 0:
                with pose_result_queue_mutex:
                    _result = pose_result_queue.popleft()
                    _ts_input, t_info, pose_results_list = _result

                _ts = time.time()
                if ts_inference is not None:
                    fps_inference = 1.0 / (_ts - ts_inference)
                ts_inference = _ts
                t_delay_inference = (_ts - _ts_input) * 1000

            # visualize detection and pose results
            if pose_results_list is not None:
                for model_info, pose_results in zip(pose_model_list,
                                                    pose_results_list):
                    pose_model = model_info['model']
                    bbox_color = model_info['bbox_color']

                    dataset_name = pose_model.cfg.data['test']['type']

                    # show pose results
                    if args.vis_mode == 1:
                        img = vis_pose_result(
                            pose_model,
                            img,
                            pose_results,
                            radius=4,
                            thickness=2,
                            dataset=dataset_name,
                            kpt_score_thr=1e7,
                            bbox_color=bbox_color)
                    elif args.vis_mode == 2:
                        img = vis_pose_result(
                            pose_model,
                            img,
                            pose_results,
                            radius=4,
                            thickness=2,
                            dataset=dataset_name,
                            kpt_score_thr=args.kpt_thr,
                            bbox_color=bbox_color)

                    # sunglasses effect
                    if args.sunglasses:
                        if dataset_name in {
                                'TopDownCocoDataset',
                                'TopDownCocoWholeBodyDataset'
                        }:
                            left_eye_idx = 1
                            right_eye_idx = 2
                        elif dataset_name == 'AnimalPoseDataset':
                            left_eye_idx = 0
                            right_eye_idx = 1
                        else:
                            raise ValueError(
                                'Sunglasses effect does not support'
                                f'{dataset_name}')
                        if sunglasses_img is None:
                            # The image attributes to:
                            # https://www.vecteezy.com/free-vector/glass
                            # Glass Vectors by Vecteezy
                            sunglasses_img = cv2.imread(
                                'demo/resources/sunglasses.jpg')
                        img = apply_sunglasses_effect(img, pose_results,
                                                      sunglasses_img,
                                                      left_eye_idx,
                                                      right_eye_idx)
                    # bug-eye effect
                    if args.bugeye:
                        if dataset_name in {
                                'TopDownCocoDataset',
                                'TopDownCocoWholeBodyDataset'
                        }:
                            left_eye_idx = 1
                            right_eye_idx = 2
                        elif dataset_name == 'AnimalPoseDataset':
                            left_eye_idx = 0
                            right_eye_idx = 1
                        else:
                            raise ValueError('Bug-eye effect does not support'
                                             f'{dataset_name}')
                        img = apply_bugeye_effect(img, pose_results,
                                                  left_eye_idx, right_eye_idx)

            # delay control
            if args.display_delay > 0:
                t_sleep = args.display_delay * 0.001 - (time.time() - ts_input)
                if t_sleep > 0:
                    time.sleep(t_sleep)
            t_delay = (time.time() - ts_input) * 1000

            # show time information
            t_info_display = stop_watch.report_strings()  # display fps
            t_info_display.append(f'Inference FPS: {fps_inference:>5.1f}')
            t_info_display.append(f'Delay: {t_delay:>3.0f}')
            t_info_display.append(
                f'Inference Delay: {t_delay_inference:>3.0f}')
            t_info_str = ' | '.join(t_info_display + t_info)
            cv2.putText(img, t_info_str, (20, 20), cv2.FONT_HERSHEY_DUPLEX,
                        0.3, text_color, 1)
            # collect system information
            sys_info = [
                f'RES: {img.shape[1]}x{img.shape[0]}',
                f'Buffer: {frame_buffer.qsize()}/{frame_buffer.maxsize}'
            ]
            if psutil_proc is not None:
                sys_info += [
                    f'CPU: {psutil_proc.cpu_percent():.1f}%',
                    f'MEM: {psutil_proc.memory_percent():.1f}%'
                ]
            sys_info_str = ' | '.join(sys_info)
            cv2.putText(img, sys_info_str, (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                        0.3, text_color, 1)

            # save the output video frame
            if args.out_video_file is not None:
                if vid_out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = args.out_video_fps
                    frame_size = (img.shape[1], img.shape[0])
                    vid_out = cv2.VideoWriter(args.out_video_file, fourcc, fps,
                                              frame_size)

                vid_out.write(img)

            # display
            cv2.imshow('mmpose webcam demo', img)
            keyboard_input = cv2.waitKey(1)
            if keyboard_input in (27, ord('q'), ord('Q')):
                break
            elif keyboard_input == ord('s'):
                args.sunglasses = not args.sunglasses
            elif keyboard_input == ord('b'):
                args.bugeye = not args.bugeye
            elif keyboard_input == ord('v'):
                args.vis_mode = (args.vis_mode + 1) % 3

    cv2.destroyAllWindows()
    if vid_out is not None:
        vid_out.release()
    event_exit.set()


def main():
    global args
    global frame_buffer
    global input_queue, input_queue_mutex
    global det_result_queue, det_result_queue_mutex
    global pose_result_queue, pose_result_queue_mutex
    global det_model, pose_model_list, pose_history_list
    global event_exit, event_inference_done

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

    # frame buffer
    if args.buffer_size > 0:
        buffer_size = args.buffer_size
    else:
        # infer buffer size from the display delay time
        # assume that the maximum video fps is 30
        buffer_size = round(30 * (1 + max(args.display_delay, 0) / 1000.))
    frame_buffer = Queue(maxsize=buffer_size)

    # queue of input frames
    # element: (timestamp, frame)
    input_queue = deque(maxlen=1)
    input_queue_mutex = Lock()

    # queue of detection results
    # element: tuple(timestamp, frame, time_info, det_results)
    det_result_queue = deque(maxlen=1)
    det_result_queue_mutex = Lock()

    # queue of detection/pose results
    # element: (timestamp, time_info, pose_results_list)
    pose_result_queue = deque(maxlen=1)
    pose_result_queue_mutex = Lock()

    try:
        event_exit = Event()
        event_inference_done = Event()
        t_input = Thread(target=read_camera, args=())
        t_det = Thread(target=inference_detection, args=(), daemon=True)
        t_pose = Thread(target=inference_pose, args=(), daemon=True)

        t_input.start()
        t_det.start()
        t_pose.start()

        # run display in the main thread
        display()
        # join the input thread (non-daemon)
        t_input.join()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
