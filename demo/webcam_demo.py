import argparse

import cv2
import numpy as np
from mmcv import Timer

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


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
    pose_models = []
    if args.enable_human_pose:
        pose_model = init_pose_model(
            args.human_pose_config,
            args.human_pose_checkpoint,
            device=args.device.lower())
        model_name = 'human'
        cat_ids = args.human_det_ids
        pose_models.append((model_name, pose_model, cat_ids))
    if args.enable_animal_pose:
        pose_model = init_pose_model(
            args.animal_pose_config,
            args.animal_pose_checkpoint,
            device=args.device.lower())
        model_name = 'animal'
        cat_ids = args.animal_det_ids
        pose_models.append((model_name, pose_model, cat_ids))

    # init video reader
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f'Cannot open camera (ID={args.cam_id})')
        exit()

    timer = Timer()
    fps_check_interval = 10  # update fps information every 10 frames
    frame_count = 0
    fps = -1

    while True:
        # capture a camera frame
        ret_val, frame = cap.read()
        if not ret_val:
            break

        # inference detection
        mmdet_results = inference_detector(det_model, frame)
        img = frame.copy()

        for model_name, pose_model, cat_ids in pose_models:
            det_results = process_mmdet_results(
                mmdet_results, class_names=det_model.CLASSES, cat_ids=cat_ids)

            dataset_name = pose_model.cfg.data['test']['type']
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                frame,
                det_results,
                bbox_thr=args.det_score_thr,
                format='xyxy',
                dataset=dataset_name)

            img = vis_pose_result(
                pose_model,
                img,
                pose_results,
                dataset=dataset_name,
                kpt_score_thr=args.kpt_thr)

        # show fps
        frame_count += 1
        if frame_count % fps_check_interval == 0:
            fps = fps_check_interval / timer.since_last_check()
        cv2.putText(img, f'Shape: {img.shape[:2]}', (30, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (242, 192, 135), 1)
        cv2.putText(img, f'FPS: {fps:.1f}', (200, 20), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (242, 192, 135), 1)

        cv2.imshow('mmpose webcam demo', img)
        if cv2.waitKey(1) == 27:  # esc to stop
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument(
        '--det_config',
        type=str,
        default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
        'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.3'
        '84_20200504_210434-a5d8aa15.pth',
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

    args = parser.parse_args()

    webcam_demo(args)
