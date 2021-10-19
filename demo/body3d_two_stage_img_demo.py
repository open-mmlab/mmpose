# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import mmcv
import numpy as np
from xtcocotools.coco import COCO

from mmpose.apis import (inference_pose_lifter_model,
                         inference_top_down_pose_model, vis_3d_pose_result)
from mmpose.apis.inference import init_pose_model
from mmpose.core import SimpleCamera
from mmpose.datasets import DatasetInfo


def _keypoint_camera_to_world(keypoints,
                              camera_params,
                              image_name=None,
                              dataset='Body3DH36MDataset'):
    """Project 3D keypoints from the camera space to the world space.

    Args:
        keypoints (np.ndarray): 3D keypoints in shape [..., 3]
        camera_params (dict): Parameters for all cameras.
        image_name (str): The image name to specify the camera.
        dataset (str): The dataset type, e.g. Body3DH36MDataset.
    """
    cam_key = None
    if dataset == 'Body3DH36MDataset':
        subj, rest = osp.basename(image_name).split('_', 1)
        _, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        cam_key = (subj, camera)
    else:
        raise NotImplementedError

    camera = SimpleCamera(camera_params[cam_key])
    keypoints_world = keypoints.copy()
    keypoints_world[..., :3] = camera.camera_to_world(keypoints[..., :3])

    return keypoints_world


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'pose_lifter_config',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        'pose_lifter_checkpoint',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--pose-detector-config',
        type=str,
        default=None,
        help='Config file for the 1st stage 2D pose detector')
    parser.add_argument(
        '--pose-detector-checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 1st stage 2D pose detector')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default=None,
        help='Json file containing image and bbox information. Optionally,'
        'The Json file can also contain 2D pose information. See'
        '"only-second-stage"')
    parser.add_argument(
        '--camera-param-file',
        type=str,
        default=None,
        help='Camera parameter file for converting 3D pose predictions from '
        ' the camera space to to world space. If None, no conversion will be '
        'applied.')
    parser.add_argument(
        '--only-second-stage',
        action='store_true',
        help='If true, load 2D pose detection result from the Json file and '
        'skip the 1st stage. The pose detection model will be ignored.')
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
        help='If True, show ground truth if it is available. The ground truth '
        'should be contained in the annotations in the Json file with the key '
        '"keypoints_3d" for each instance.')
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
    parser.add_argument('--kpt-thr', type=float, default=0.3)
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

    # First stage: 2D pose detection
    pose_det_results_list = []
    if args.only_second_stage:
        from mmpose.apis.inference import _xywh2xyxy

        print('Stage 1: load 2D pose results from Json file.')
        for image_id, image in coco.imgs.items():
            image_name = osp.join(args.img_root, image['file_name'])
            ann_ids = coco.getAnnIds(image_id)
            pose_det_results = []
            for ann_id in ann_ids:
                ann = coco.anns[ann_id]
                keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                keypoints[..., 2] = keypoints[..., 2] >= 1
                keypoints_3d = np.array(ann['keypoints_3d']).reshape(-1, 4)
                keypoints_3d[..., 3] = keypoints_3d[..., 3] >= 1
                bbox = np.array(ann['bbox']).reshape(1, -1)

                pose_det_result = {
                    'image_name': image_name,
                    'bbox': _xywh2xyxy(bbox),
                    'keypoints': keypoints,
                    'keypoints_3d': keypoints_3d
                }
                pose_det_results.append(pose_det_result)
            pose_det_results_list.append(pose_det_results)

    else:
        print('Stage 1: 2D pose detection.')

        pose_det_model = init_pose_model(
            args.pose_detector_config,
            args.pose_detector_checkpoint,
            device=args.device.lower())

        assert pose_det_model.cfg.model.type == 'TopDown', 'Only "TopDown"' \
            'model is supported for the 1st stage (2D pose detection)'

        dataset = pose_det_model.cfg.data['test']['type']
        dataset_info = pose_det_model.cfg.data['test'].get(
            'dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)

        img_keys = list(coco.imgs.keys())

        for i in mmcv.track_iter_progress(range(len(img_keys))):
            # get bounding box annotations
            image_id = img_keys[i]
            image = coco.loadImgs(image_id)[0]
            image_name = osp.join(args.img_root, image['file_name'])
            ann_ids = coco.getAnnIds(image_id)

            # make person results for single image
            person_results = []
            for ann_id in ann_ids:
                person = {}
                ann = coco.anns[ann_id]
                person['bbox'] = ann['bbox']
                person_results.append(person)

            pose_det_results, _ = inference_top_down_pose_model(
                pose_det_model,
                image_name,
                person_results,
                bbox_thr=None,
                format='xywh',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None)

            for res in pose_det_results:
                res['image_name'] = image_name
            pose_det_results_list.append(pose_det_results)

    # Second stage: Pose lifting
    print('Stage 2: 2D-to-3D pose lifting.')

    pose_lift_model = init_pose_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert pose_lift_model.cfg.model.type == 'PoseLifter', 'Only' \
        '"PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'
    dataset = pose_lift_model.cfg.data['test']['type']
    dataset_info = pose_lift_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    camera_params = None
    if args.camera_param_file is not None:
        camera_params = mmcv.load(args.camera_param_file)

    for i, pose_det_results in enumerate(
            mmcv.track_iter_progress(pose_det_results_list)):
        # 2D-to-3D pose lifting
        # Note that the pose_det_results are regarded as a single-frame pose
        # sequence
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=[pose_det_results],
            dataset=dataset,
            dataset_info=dataset_info,
            with_track_id=False)

        image_name = pose_det_results[0]['image_name']

        # Pose processing
        pose_lift_results_vis = []
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d']
            # project to world space
            if camera_params is not None:
                keypoints_3d = _keypoint_camera_to_world(
                    keypoints_3d,
                    camera_params=camera_params,
                    image_name=image_name,
                    dataset=dataset)
            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # Add title
            det_res = pose_det_results[idx]
            instance_id = det_res.get('track_id', idx)
            res['title'] = f'Prediction ({instance_id})'
            pose_lift_results_vis.append(res)
            # Add ground truth
            if args.show_ground_truth:
                if 'keypoints_3d' not in det_res:
                    print('Fail to show ground truth. Please make sure that'
                          ' the instance annotations from the Json file'
                          ' contain "keypoints_3d".')
                else:
                    gt = res.copy()
                    gt['keypoints_3d'] = det_res['keypoints_3d']
                    gt['title'] = f'Ground truth ({instance_id})'
                    pose_lift_results_vis.append(gt)

        # Visualization
        if args.out_img_root is None:
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = osp.join(args.out_img_root, f'vis_{i}.jpg')

        vis_3d_pose_result(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=image_name,
            dataset_info=dataset_info,
            out_file=out_file)


if __name__ == '__main__':
    main()
