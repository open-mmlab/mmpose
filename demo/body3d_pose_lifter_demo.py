# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import cv2
import mmcv
import numpy as np
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmpose.apis import (_track_by_iou, _track_by_oks, collect_multi_frames,
                         extract_pose_sequence, inference_pose_lifter_model,
                         inference_topdown, init_model)
from mmpose.models.pose_estimators import PoseLifter
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample, merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def convert_keypoint_definition(keypoints, pose_det_dataset,
                                pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition, so that they are compatible with the definitions
    required for 3D pose lifting.

    Args:
        keypoints (ndarray[N, K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.

    Returns:
        ndarray[K, 2 or 3]: the transformed 2D keypoints.
    """
    assert pose_lift_dataset in [
        'Human36mDataset'], '`pose_lift_dataset` should be ' \
        f'`Human36mDataset`, but got {pose_lift_dataset}.'

    coco_style_datasets = [
        'CocoDataset', 'PoseTrack18VideoDataset', 'PoseTrack18Dataset'
    ]
    keypoints_new = np.zeros((keypoints.shape[0], 17, keypoints.shape[2]),
                             dtype=keypoints.dtype)
    if pose_lift_dataset == 'Human36mDataset':
        if pose_det_dataset in ['Human36mDataset']:
            keypoints_new = keypoints
        elif pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[:, 0] = (keypoints[:, 11] + keypoints[:, 12]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[:, 8] = (keypoints[:, 5] + keypoints[:, 6]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[:,
                          7] = (keypoints_new[:, 0] + keypoints_new[:, 8]) / 2
            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[:, 10] = (keypoints[:, 1] + keypoints[:, 2]) / 2
            # rearrange other keypoints
            keypoints_new[:, [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
                keypoints[:, [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        elif pose_det_dataset in ['AicDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[:, 0] = (keypoints[:, 9] + keypoints[:, 6]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[:, 8] = (keypoints[:, 3] + keypoints[:, 0]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[:,
                          7] = (keypoints_new[:, 0] + keypoints_new[:, 8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[:, 9] = (3 * keypoints[:, 13] + keypoints[:, 12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[:, 10] = (5 * keypoints[:, 13] +
                                    7 * keypoints[:, 12]) / 12

            keypoints_new[:, [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[:, [6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]]
        elif pose_det_dataset in ['CrowdPoseDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[:, 0] = (keypoints[:, 6] + keypoints[:, 7]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[:, 8] = (keypoints[:, 0] + keypoints[:, 1]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[:,
                          7] = (keypoints_new[:, 0] + keypoints_new[:, 8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[:, 9] = (3 * keypoints[:, 13] + keypoints[:, 12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[:, 10] = (5 * keypoints[:, 13] +
                                    7 * keypoints[:, 12]) / 12

            keypoints_new[:, [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[:, [7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]]
        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    return keypoints_new


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose_estimator_config',
        type=str,
        default=None,
        help='Config file for the 1st stage 2D pose estimator')
    parser.add_argument(
        'pose_estimator_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 1st stage 2D pose estimator')
    parser.add_argument(
        'pose_lifter_config',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        'pose_lifter_checkpoint',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument('--input', type=str, default='', help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to show visualizations')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--norm-pose-2d',
        action='store_true',
        help='Scale the bbox (along with the 2D pose) to the average bbox '
        'scale of the dataset, and move the bbox (along with the 2D pose) to '
        'the average bbox center of the dataset. This is useful when bbox '
        'is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
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
        default=0.9,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
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
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the 2D pose'
        'detection stage. Default: False.')

    args = parser.parse_args()
    return args


def get_area(results):
    for i, data_sample in enumerate(results):
        pred_instance = data_sample.pred_instances.cpu().numpy()
        if 'bboxes' in pred_instance:
            bboxes = pred_instance.bboxes
            results[i].pred_instances.set_field(
                np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                          for bbox in bboxes]), 'areas')
        else:
            keypoints = pred_instance.keypoints
            areas, bboxes = [], []
            for keypoint in keypoints:
                xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                xmax = np.max(keypoint[:, 0])
                ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                ymax = np.max(keypoint[:, 1])
                areas.append((xmax - xmin) * (ymax - ymin))
                bboxes.append([xmin, ymin, xmax, ymax])
            results[i].pred_instances.areas = np.array(areas)
            results[i].pred_instances.bboxes = np.array(bboxes)
    return results


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
        args.pose_estimator_config,
        args.pose_estimator_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_estimator, TopdownPoseEstimator), 'Only "TopDown"' \
        'model is supported for the 1st stage (2D pose detection)'

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get(
        'skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get(
        'skeleton_link_colors', None)

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices' in pose_estimator.cfg.test_dataloader.dataset
        indices = pose_estimator.cfg.test_dataloader.dataset[
            'frame_indices_test']

    pose_det_dataset = pose_estimator.cfg.test_dataloader.dataset

    pose_lifter = init_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_lifter, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'
    pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset

    pose_lifter.cfg.visualizer.radius = args.radius
    pose_lifter.cfg.visualizer.line_width = args.thickness
    local_visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)

    # the dataset_meta is loaded from the checkpoint
    local_visualizer.set_dataset_meta(pose_lifter.dataset_meta)

    init_default_scope(pose_lifter.cfg.get('default_scope', 'mmpose'))

    if args.output_root == '':
        save_out_video = False
    else:
        os.makedirs(args.output_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None

    pose_est_results_list = []
    next_id = 0
    pose_est_results = []

    video = cv2.VideoCapture(args.input)
    assert video.isOpened(), f'Failed to load video file {args.input}'

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        width = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_idx = -1

    while video.isOpened():
        success, frame = video.read()
        frame_idx += 1

        if not success:
            break

        pose_est_results_last = pose_est_results

        # First stage: 2D pose detection
        # test a single image, the resulting box is (x1, y1, x2, y2)
        det_result = inference_detector(detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()

        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                       pred_instance.scores > args.bbox_thr)]

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_idx, indices,
                                          args.online)

        # make person results for current image
        pose_est_results = inference_topdown(
            pose_estimator, frames if args.use_multi_frames else frame, bboxes)

        pose_est_results = get_area(pose_est_results)
        if args.use_oks_tracking:
            _track = partial(_track_by_oks)
        else:
            _track = _track_by_iou

        for i, result in enumerate(pose_est_results):
            track_id, pose_est_results_last, match_result = _track(
                result, pose_est_results_last, args.tracking_thr)
            if track_id == -1:
                pred_instances = result.pred_instances.cpu().numpy()
                keypoints = pred_instances.keypoints
                if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    pose_est_results[i].set_field(next_id, 'track_id')
                    next_id += 1
                else:
                    # If the number of keypoints detected is small,
                    # delete that person instance.
                    keypoints[:, :, 1] = -10
                    pose_est_results[i].pred_instances.set_field(
                        keypoints, 'keypoints')
                    bboxes = pred_instances.bboxes * 0
                    pose_est_results[i].pred_instances.set_field(
                        bboxes, 'bboxes')
                    pose_est_results[i].set_field(-1, 'track_id')
                    pose_est_results[i].set_field(pred_instances,
                                                  'pred_instances')
            else:
                pose_est_results[i].set_field(track_id, 'track_id')

            del match_result

        pose_est_results_converted = []
        for pose_est_result in pose_est_results:
            pose_est_result_converted = PoseDataSample()
            gt_instances = InstanceData()
            pred_instances = InstanceData()
            for k in pose_est_result.gt_instances.keys():
                gt_instances.set_field(pose_est_result.gt_instances[k], k)
            for k in pose_est_result.pred_instances.keys():
                pred_instances.set_field(pose_est_result.pred_instances[k], k)
            pose_est_result_converted.gt_instances = gt_instances
            pose_est_result_converted.pred_instances = pred_instances
            pose_est_result_converted.track_id = pose_est_result.track_id
            pose_est_results_converted.append(pose_est_result_converted)

        for i, result in enumerate(pose_est_results_converted):
            keypoints = result.pred_instances.keypoints
            keypoints = convert_keypoint_definition(keypoints,
                                                    pose_det_dataset['type'],
                                                    pose_lift_dataset['type'])
            pose_est_results_converted[i].pred_instances.keypoints = keypoints

        pose_est_results_list.append(pose_est_results_converted.copy())

        # extract and pad input pose2d sequence
        pose_results_2d = extract_pose_sequence(
            pose_est_results_list,
            frame_idx=frame_idx,
            causal=pose_lifter.causal,
            seq_len=pose_lift_dataset.get('seq_len', 1),
            step=pose_lift_dataset.get('seq_step', 1))

        # Second stage: Pose lifting
        # 2D-to-3D pose lifting
        pose_lift_results = inference_pose_lifter_model(
            pose_lifter,
            pose_results_2d,
            image_size=(width, height),
            norm_pose_2d=args.norm_pose_2d)

        # Pose processing
        for idx, pose_lift_res in enumerate(pose_lift_results):
            gt_instances = pose_lift_res.gt_instances

            pose_lift_res.track_id = pose_est_results_converted[i].get(
                'track_id', 1e4)

            pred_instances = pose_lift_res.pred_instances
            keypoints = pred_instances.keypoints

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[i].pred_instances.keypoints = keypoints

        pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        pred_3d_data_samples = merge_data_samples(pose_lift_results)

        # Visualization
        frame = mmcv.bgr2rgb(frame)

        det_data_sample = merge_data_samples(pose_est_results)

        local_visualizer.add_datasample(
            'result',
            frame,
            data_sample=pred_3d_data_samples,
            det_data_sample=det_data_sample,
            draw_gt=False,
            det_kpt_color=det_kpt_color,
            det_dataset_skeleton=det_dataset_skeleton,
            det_dataset_link_color=det_dataset_link_color,
            show=args.show,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            wait_time=0.001)

        frame_vis = local_visualizer.get_image()

        if save_out_video:
            if video_writer is None:
                # the size of the image with visualization may vary
                # depending on the presence of heatmaps
                video_writer = cv2.VideoWriter(
                    osp.join(args.output_root,
                             f'vis_{osp.basename(args.input)}'), fourcc, fps,
                    (frame_vis.shape[1], frame_vis.shape[0]))

            video_writer.write(mmcv.rgb2bgr(frame_vis))

    video.release()

    if video_writer:
        video_writer.release()


if __name__ == '__main__':
    main()
