import argparse

import cv2
import mmcv
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


def webcam_demo(args):
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

    # load resource
    sunglasses_img = None
    if args.sunglasses:
        # The image attributes to:
        # https://www.vecteezy.com/free-vector/glass
        # Glass Vectors by Vecteezy
        sunglasses_img = mmcv.imread('demo/resources/sunglasses.jpg')

    # init video reader and writer
    vid_cap = cv2.VideoCapture(args.cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={args.cam_id})')
        exit()
    vid_out = None

    # use stop_watch to measure time consuming
    stop_watch = StopWatch(window=10)
    while True:
        with stop_watch.timeit():
            # capture a camera frame
            ret_val, frame = vid_cap.read()
            if not ret_val:
                break

            # inference detection
            with stop_watch.timeit('Det'):
                mmdet_results = inference_detector(det_model, frame)
            img = frame.copy()

            for model_info, pose_history in zip(pose_model_list,
                                                pose_history_list):
                model_name = model_info['name']
                pose_model = model_info['model']
                cat_ids = model_info['cat_ids']
                bbox_color = model_info['bbox_color']
                pose_results_last = pose_history['pose_results_last']
                next_id = pose_history['next_id']

                with stop_watch.timeit(model_name):
                    det_results = process_mmdet_results(
                        mmdet_results,
                        class_names=det_model.CLASSES,
                        cat_ids=cat_ids)

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
                    pose_history['pose_results_last'] = pose_results
                    pose_history['next_id'] = next_id

                if args.sunglasses:
                    if dataset_name == 'TopDownCocoDataset':
                        leye_idx = 1
                        reye_idx = 2
                    elif dataset_name == 'AnimalPoseDataset':
                        leye_idx = 0
                        reye_idx = 1
                    else:
                        raise ValueError('Sunglasses effect does not support'
                                         f'{dataset_name}')
                    img = apply_sunglasses_effect(img, pose_results,
                                                  sunglasses_img, leye_idx,
                                                  reye_idx)
                elif args.bugeye:
                    if dataset_name == 'TopDownCocoDataset':
                        leye_idx = 1
                        reye_idx = 2
                    elif dataset_name == 'AnimalPoseDataset':
                        leye_idx = 0
                        reye_idx = 1
                    else:
                        raise ValueError('Bug-eye effect does not support'
                                         f'{dataset_name}')
                    img = apply_bugeye_effect(img, pose_results, leye_idx,
                                              reye_idx)
                else:
                    img = vis_pose_result(
                        pose_model,
                        img,
                        pose_results,
                        radius=4,
                        thickness=2,
                        dataset=dataset_name,
                        kpt_score_thr=args.kpt_thr,
                        bbox_color=bbox_color)

            str_info = [f'Shape: {img.shape[:2]}']
            str_info += stop_watch.report_strings()
            if psutil_proc is not None:
                str_info += [
                    f'CPU({psutil_proc.cpu_num()}): '
                    f'{psutil_proc.cpu_percent():.1f}%'
                ]
                str_info += [f'MEM: {psutil_proc.memory_percent():.1f}%']
            str_info = ' | '.join(str_info)
            cv2.putText(img, str_info, (30, 20), cv2.FONT_HERSHEY_DUPLEX, 0.3,
                        (228, 183, 61), 1)

            if args.out_video_file is not None:
                if vid_out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 20
                    frame_size = (img.shape[1], img.shape[0])
                    vid_out = cv2.VideoWriter(args.out_video_file, fourcc, fps,
                                              frame_size)

                vid_out.write(img)

            cv2.imshow('mmpose webcam demo', img)
            if cv2.waitKey(1) == 27:  # esc to stop
                break

    vid_cap.release()
    if vid_out is not None:
        vid_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
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
    args = parser.parse_args()

    webcam_demo(args)
