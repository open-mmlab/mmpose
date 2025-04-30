# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from typing import List

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.visualization import Pose3dLocalVisualizer

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from rtmpose3d import *  # noqa: F401, F403


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose3d_estimator_config',
        type=str,
        default=None,
        help='Config file for the 3D pose estimator')
    parser.add_argument(
        'pose3d_estimator_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 3D pose estimator')
    parser.add_argument('--input', type=str, default='', help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to show visualizations')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        default=False,
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='Whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.5,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='Inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')

    args = parser.parse_args()
    return args


def process_one_image(args, detector, frame: np.ndarray, frame_idx: int,
                      pose_estimator,
                      pose_est_results_last: List[PoseDataSample],
                      pose_est_results_list: List[List[PoseDataSample]],
                      next_id: int, visualize_frame: np.ndarray,
                      visualizer: Pose3dLocalVisualizer):
    """Visualize detected and predicted keypoints of one image.

    Pipeline of this function:

                              frame
                                |
                                V
                        +-----------------+
                        |     detector    |
                        +-----------------+
                                |  det_result
                                V
                        +-----------------+
                        |  pose_estimator |
                        +-----------------+
                                |  pose_est_results
                                V
                       +-----------------+
                       | post-processing |
                       +-----------------+
                                |  pred_3d_data_samples
                                V
                         +------------+
                         | visualizer |
                         +------------+

    Args:
        args (Argument): Custom command-line arguments.
        detector (mmdet.BaseDetector): The mmdet detector.
        frame (np.ndarray): The image frame read from input image or video.
        frame_idx (int): The index of current frame.
        pose_estimator (TopdownPoseEstimator): The pose estimator for 2d pose.
        pose_est_results_last (list(PoseDataSample)): The results of pose
            estimation from the last frame for tracking instances.
        pose_est_results_list (list(list(PoseDataSample))): The list of all
            pose estimation results converted by
            ``convert_keypoint_definition`` from previous frames. In
            pose-lifting stage it is used to obtain the 2d estimation sequence.
        next_id (int): The next track id to be used.
        pose_lifter (PoseLifter): The pose-lifter for estimating 3d pose.
        visualize_frame (np.ndarray): The image for drawing the results on.
        visualizer (Visualizer): The visualizer for visualizing the 2d and 3d
            pose estimation results.

    Returns:
        pose_est_results (list(PoseDataSample)): The pose estimation result of
            the current frame.
        pose_est_results_list (list(list(PoseDataSample))): The list of all
            converted pose estimation results until the current frame.
        pred_3d_instances (InstanceData): The result of pose-lifting.
            Specifically, the predicted keypoints and scores are saved at
            ``pred_3d_instances.keypoints`` and
            ``pred_3d_instances.keypoint_scores``.
        next_id (int): The next track id to be used.
    """
    # pose_dataset = pose_estimator.cfg.test_dataloader.dataset
    pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']

    # First stage: conduct 2D pose detection in a Topdown manner
    # use detector to obtain person bounding boxes
    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()

    # filter out the person instances with category and bbox threshold
    # e.g. 0 for person in COCO
    bboxes = pred_instance.bboxes
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]

    # estimate pose results for current image
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)

    # post-processing
    for idx, pose_est_result in enumerate(pose_est_results):
        pose_est_result.track_id = pose_est_results[idx].get('track_id', 1e4)

        pred_instances = pose_est_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_est_results[
                idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = -keypoints[..., [0, 2, 1]]

        # rebase height (z-axis)
        if not args.disable_rebase_keypoint:
            keypoints[..., 2] -= np.min(
                keypoints[..., 2], axis=-1, keepdims=True)

        pose_est_results[idx].pred_instances.keypoints = keypoints

    pose_est_results = sorted(
        pose_est_results, key=lambda x: x.get('track_id', 1e4))

    pred_3d_data_samples = merge_data_samples(pose_est_results)
    pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

    if args.num_instances < 0:
        args.num_instances = len(pose_est_results)

    # Visualization
    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            visualize_frame,
            data_sample=pred_3d_data_samples,
            det_data_sample=pred_3d_data_samples,
            draw_gt=False,
            draw_2d=True,
            dataset_2d=pose_det_dataset_name,
            dataset_3d=pose_det_dataset_name,
            show=args.show,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            convert_keypoint=False,
            axis_limit=400,
            axis_azimuth=70,
            axis_elev=15,
            num_instances=args.num_instances,
            wait_time=args.show_interval)

    return pose_est_results, pose_est_results_list, pred_3d_instances, next_id


def main():
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_model(
        args.pose3d_estimator_config,
        args.pose3d_estimator_checkpoint,
        device=args.device.lower())

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get(
        'skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get(
        'skeleton_link_colors', None)

    pose_estimator.cfg.model.test_cfg.mode = 'vis'
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.line_width = args.thickness
    pose_estimator.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_estimator.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.det_dataset_link_color = det_dataset_link_color  # noqa: E501
    pose_estimator.cfg.visualizer.skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.link_color = det_dataset_link_color
    pose_estimator.cfg.visualizer.kpt_color = det_kpt_color
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if args.output_root == '':
        save_output = False
    else:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'
        save_output = True

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    pose_est_results_list = []
    pred_instances_list = []
    if input_type == 'image':
        frame = mmcv.imread(args.input, channel_order='rgb')
        _, _, pred_3d_instances, _ = process_one_image(
            args=args,
            detector=detector,
            frame=args.input,
            frame_idx=0,
            pose_estimator=pose_estimator,
            pose_est_results_last=[],
            pose_est_results_list=pose_est_results_list,
            next_id=0,
            visualize_frame=frame,
            visualizer=visualizer)

        if args.save_predictions:
            # save prediction results
            pred_instances_list = split_instances(pred_3d_instances)

        if save_output:
            frame_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(frame_vis), output_file)

    elif input_type in ['webcam', 'video']:
        next_id = 0
        pose_est_results = []

        if args.input == 'webcam':
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(args.input)

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)

        video_writer = None
        frame_idx = 0

        while video.isOpened():
            success, frame = video.read()
            frame_idx += 1

            if not success:
                break

            pose_est_results_last = pose_est_results

            # First stage: 2D pose detection
            # make person results for current image
            (pose_est_results, pose_est_results_list, pred_3d_instances,
             next_id) = process_one_image(
                 args=args,
                 detector=detector,
                 frame=frame,
                 frame_idx=frame_idx,
                 pose_estimator=pose_estimator,
                 pose_est_results_last=pose_est_results_last,
                 pose_est_results_list=pose_est_results_list,
                 next_id=next_id,
                 visualize_frame=mmcv.bgr2rgb(frame),
                 visualizer=visualizer)

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_3d_instances)))

            if save_output:
                frame_vis = visualizer.get_image()
                if video_writer is None:
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(output_file, fourcc, fps,
                                                   (frame_vis.shape[1],
                                                    frame_vis.shape[0]))
                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                time.sleep(args.show_interval)

        video.release()

        if video_writer:
            video_writer.release()
    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if save_output:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
