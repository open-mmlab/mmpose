import copy
import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np

from mmpose.apis import (get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, vis_3d_pose_result)
from mmpose.apis.inference import init_pose_model

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    Args:
        mmdet_results (list|tuple): mmdet results.
        cat_id (int): category id (default: 1 for human)

    Returns:
        person_results (list): a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def dataset_transform(keypoints, pose_det_dataset, pose_lift_dataset):
    """Transform pose det dataset keypoints convention to pose lifter dataset
    keypoints convention.

    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.
    """
    if pose_det_dataset == 'TopDownH36MDataset' and \
            pose_lift_dataset == 'Body3DH36MDataset':
        return keypoints
    elif pose_det_dataset == 'TopDownCocoDataset' and \
            pose_lift_dataset == 'Body3DH36MDataset':
        keypoints_new = np.zeros((17, 2))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11, :2] + keypoints[12, :2]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5, :2] + keypoints[6, :2]) / 2
        # head is in the middle of l_eye and r_eye
        keypoints_new[10] = (keypoints[1, :2] + keypoints[2, :2]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0, :2] + keypoints_new[8, :2]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10], :2]
        if keypoints.shape[-1] == 3:
            keypoints_new = np.concatenate([keypoints_new, keypoints[:, 2:3]],
                                           axis=1)
        return keypoints_new
    else:
        raise NotImplementedError


def main():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose_detector_config',
        type=str,
        default=None,
        help='Config file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_detector_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_lifter_config',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        'pose_lifter_checkpoint',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--video-path', type=str, default='', help='Video path')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        type=str,
        default=None,
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
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
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Failed to load video file {args.video_path}'
    num_frames = video.frame_cnt

    # First stage: 2D pose detection
    print('Stage 1: 2D pose detection.')

    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    pose_det_model = init_pose_model(
        args.pose_detector_config,
        args.pose_detector_checkpoint,
        device=args.device.lower())

    assert pose_det_model.cfg.model.type == 'TopDown', 'Only "TopDown"' \
        'model is supported for the 1st stage (2D pose detection)'

    pose_det_dataset = pose_det_model.cfg.data['test']['type']

    pose_det_results_list = []
    next_id = 0
    pose_det_results = []
    for frame in video:
        pose_det_results_last = pose_det_results

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, frame)

        # keep the person class bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results,
                                                   args.det_cat_id)

        # make person results for single image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            frame,
            person_det_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=pose_det_dataset,
            return_heatmap=False,
            outputs=None)

        # get track id for each person instance
        pose_det_results, next_id = get_track_id(
            pose_det_results,
            pose_det_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr,
            use_one_euro=args.euro,
            fps=video.fps)

        pose_det_results_list.append(copy.deepcopy(pose_det_results))

    # Second stage: Pose lifting
    print('Stage 2: 2D-to-3D pose lifting.')

    pose_lift_model = init_pose_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert pose_lift_model.cfg.model.type == 'PoseLifter', \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.fps
        writer = None

    # transform the predicted 2D keypoints to the keypoints convention of pose
    # lift dataset
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = dataset_transform(keypoints, pose_det_dataset,
                                                 pose_lift_dataset)

    # prepare for temporal padding
    seq_len = pose_lift_model.cfg.test_data_cfg.seq_len
    causal = pose_lift_model.cfg.test_data_cfg.causal
    seq_frame_interval = pose_lift_model.cfg.test_data_cfg.seq_frame_interval
    _step = seq_frame_interval
    if causal:
        frames_left = seq_len - 1
        frames_right = 0
    else:
        frames_left = (seq_len - 1) // 2
        frames_right = frames_left
    target_frame_idx = -1 if causal else seq_len // 2

    for i in mmcv.track_iter_progress(range(num_frames)):
        # get the padded sequence
        pad_left = max(0, frames_left - i // _step)
        pad_right = max(0, frames_right - (num_frames - 1 - i) // _step)
        start = max(i % _step, i - frames_left * _step)
        end = min(num_frames - (num_frames - 1 - i) % _step,
                  i + frames_right * _step + 1)
        pose_det_results_seq = [pose_det_results_list[0]] * pad_left + \
            pose_det_results_list[start:end:_step] + \
            [pose_det_results_list[-1]] * pad_right

        # 2D-to-3D pose lifting
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=pose_det_results_seq,
            dataset=pose_lift_dataset,
            with_track_id=True,
            target_frame=target_frame_idx,
            image_size=video.resolution)

        # Pose processing
        pose_lift_results_vis = []
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d']
            # exchange y-axis and z-axis, and then reverse the z-axis direction
            keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            keypoints_3d[..., 0] = -keypoints_3d[..., 0]
            keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # add title
            det_res = pose_det_results_list[i][idx]
            instance_id = det_res['track_id']
            res['title'] = f'Prediction ({instance_id})'
            pose_lift_results_vis.append(res)
            # only visualize the target frame
            keypoints_2d = res['keypoints']
            res['keypoints'] = keypoints_2d[target_frame_idx]
            res['bbox'] = det_res['bbox']

        # Visualization
        img_vis = vis_3d_pose_result(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=video[i],
            out_file=None,
            radius=args.radius,
            thickness=args.thickness)

        if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    osp.join(args.out_video_root,
                             f'vis_{osp.basename(args.video_path)}'), fourcc,
                    fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis)

    if save_out_video:
        writer.release()


if __name__ == '__main__':
    main()
