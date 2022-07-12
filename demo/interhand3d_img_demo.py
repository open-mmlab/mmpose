# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser

import mmcv
import numpy as np
from xtcocotools.coco import COCO

from mmpose.apis import inference_interhand_3d_model, vis_3d_pose_result
from mmpose.apis.inference import init_pose_model
from mmpose.core import SimpleCamera


def _transform_interhand_camera_param(interhand_camera_param):
    """Transform the camera parameters in interhand2.6m dataset to the format
    of SimpleCamera.

    Args:
        interhand_camera_param (dict): camera parameters including:
            - camrot: 3x3, camera rotation matrix (world-to-camera)
            - campos: 3x1, camera location in world space
            - focal: 2x1, camera focal length
            - princpt: 2x1, camera center

    Returns:
        param (dict): camera parameters including:
            - R: 3x3, camera rotation matrix (camera-to-world)
            - T: 3x1, camera translation (camera-to-world)
            - f: 2x1, camera focal length
            - c: 2x1, camera center
    """
    camera_param = {}
    camera_param['R'] = np.array(interhand_camera_param['camrot']).T
    camera_param['T'] = np.array(interhand_camera_param['campos'])[:, None]
    camera_param['f'] = np.array(interhand_camera_param['focal'])[:, None]
    camera_param['c'] = np.array(interhand_camera_param['princpt'])[:, None]
    return camera_param


def main():
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose network')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--camera-param-file',
        type=str,
        default=None,
        help='Camera parameter file for converting 3D pose predictions from '
        ' the pixel space to camera space. If None, keypoints in pixel space'
        'will be visualized')
    parser.add_argument(
        '--gt-joints-file',
        type=str,
        default=None,
        help='Optional argument. Ground truth 3D keypoint parameter file. '
        'If None, gt keypoints will not be shown and keypoints in pixel '
        'space will be visualized.')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--show-ground-truth',
        action='store_true',
        help='If True, show ground truth keypoint if it is available.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default=None,
        help='Root of the output visualization images. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()
    assert args.show or (args.out_img_root != '')

    coco = COCO(args.json_file)

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']

    # load camera parameters
    camera_params = None
    if args.camera_param_file is not None:
        camera_params = mmcv.load(args.camera_param_file)
    # load ground truth joints parameters
    gt_joint_params = None
    if args.gt_joints_file is not None:
        gt_joint_params = mmcv.load(args.gt_joints_file)

    # load hand bounding boxes
    det_results_list = []
    for image_id, image in coco.imgs.items():
        image_name = osp.join(args.img_root, image['file_name'])

        ann_ids = coco.getAnnIds(image_id)
        det_results = []

        capture_key = str(image['capture'])
        camera_key = image['camera']
        frame_idx = image['frame_idx']

        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            if camera_params is not None:
                camera_param = {
                    key: camera_params[capture_key][key][camera_key]
                    for key in camera_params[capture_key].keys()
                }
                camera_param = _transform_interhand_camera_param(camera_param)
            else:
                camera_param = None
            if gt_joint_params is not None:
                joint_param = gt_joint_params[capture_key][str(frame_idx)]
                gt_joint = np.concatenate([
                    np.array(joint_param['world_coord']),
                    np.array(joint_param['joint_valid'])
                ],
                                          axis=-1)
            else:
                gt_joint = None

            det_result = {
                'image_name': image_name,
                'bbox': ann['bbox'],  # bbox format is 'xywh'
                'camera_param': camera_param,
                'keypoints_3d_gt': gt_joint
            }
            det_results.append(det_result)
        det_results_list.append(det_results)

    for i, det_results in enumerate(
            mmcv.track_iter_progress(det_results_list)):

        image_name = det_results[0]['image_name']

        pose_results = inference_interhand_3d_model(
            pose_model, image_name, det_results, dataset=dataset)

        # Post processing
        pose_results_vis = []
        for idx, res in enumerate(pose_results):
            keypoints_3d = res['keypoints_3d']
            # normalize kpt score
            if keypoints_3d[:, 3].max() > 1:
                keypoints_3d[:, 3] /= 255
            # get 2D keypoints in pixel space
            res['keypoints'] = keypoints_3d[:, [0, 1, 3]]

            # For model-predicted keypoints, channel 0 and 1 are coordinates
            # in pixel space, and channel 2 is the depth (in mm) relative
            # to root joints.
            # If both camera parameter and absolute depth of root joints are
            # provided, we can transform keypoint to camera space for better
            # visualization.
            camera_param = res['camera_param']
            keypoints_3d_gt = res['keypoints_3d_gt']
            if camera_param is not None and keypoints_3d_gt is not None:
                # build camera model
                camera = SimpleCamera(camera_param)
                # transform gt joints from world space to camera space
                keypoints_3d_gt[:, :3] = camera.world_to_camera(
                    keypoints_3d_gt[:, :3])

                # transform relative depth to absolute depth
                keypoints_3d[:21, 2] += keypoints_3d_gt[20, 2]
                keypoints_3d[21:, 2] += keypoints_3d_gt[41, 2]

                # transform keypoints from pixel space to camera space
                keypoints_3d[:, :3] = camera.pixel_to_camera(
                    keypoints_3d[:, :3])

            # rotate the keypoint to make z-axis correspondent to height
            # for better visualization
            vis_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            keypoints_3d[:, :3] = keypoints_3d[:, :3] @ vis_R
            if keypoints_3d_gt is not None:
                keypoints_3d_gt[:, :3] = keypoints_3d_gt[:, :3] @ vis_R

            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                valid = keypoints_3d[..., 3] > 0
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[valid, 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            res['keypoints_3d_gt'] = keypoints_3d_gt

            # Add title
            instance_id = res.get('track_id', idx)
            res['title'] = f'Prediction ({instance_id})'
            pose_results_vis.append(res)
            # Add ground truth
            if args.show_ground_truth:
                if keypoints_3d_gt is None:
                    print('Fail to show ground truth. Please make sure that'
                          ' gt-joints-file is provided.')
                else:
                    gt = res.copy()
                    if args.rebase_keypoint_height:
                        valid = keypoints_3d_gt[..., 3] > 0
                        keypoints_3d_gt[..., 2] -= np.min(
                            keypoints_3d_gt[valid, 2], axis=-1, keepdims=True)
                    gt['keypoints_3d'] = keypoints_3d_gt
                    gt['title'] = f'Ground truth ({instance_id})'
                    pose_results_vis.append(gt)

        # Visualization
        if args.out_img_root is None:
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = osp.join(args.out_img_root, f'vis_{i}.jpg')

        vis_3d_pose_result(
            pose_model,
            result=pose_results_vis,
            img=det_results[0]['image_name'],
            out_file=out_file,
            dataset=dataset,
            show=args.show,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            axis_azimuth=-115,
        )


if __name__ == '__main__':
    main()
