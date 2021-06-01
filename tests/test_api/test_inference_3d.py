import numpy as np
import pytest
import torch

from mmpose.apis import (inference_pose_lifter_model, init_pose_model,
                         vis_3d_pose_result)


def test_pose_lifter_demo():
    # H36M demo
    pose_model = init_pose_model(
        'configs/body/3d_kpt_sview_rgb_img/pose_lift/'
        'h36m/simplebaseline3d_h36m.py',
        None,
        device='cpu')

    pose_det_result = {
        'keypoints': np.zeros((17, 3)),
        'bbox': [50, 50, 50, 50],
        'track_id': 0,
        'image_name': 'tests/data/h36m/S1_Directions_1.54138969_000001.jpg',
    }

    pose_results_2d = [[pose_det_result]]

    dataset = pose_model.cfg.data['test']['type']

    _ = inference_pose_lifter_model(
        pose_model, pose_results_2d, dataset, with_track_id=False)

    pose_lift_results = inference_pose_lifter_model(
        pose_model, pose_results_2d, dataset, with_track_id=True)

    for res in pose_lift_results:
        res['title'] = 'title'
    vis_3d_pose_result(
        pose_model,
        pose_lift_results,
        img=pose_lift_results[0]['image_name'],
        dataset=dataset)

    # test special cases
    # Empty 2D results
    _ = inference_pose_lifter_model(
        pose_model, [[]], dataset, with_track_id=False)

    if torch.cuda.is_available():
        _ = inference_pose_lifter_model(
            pose_model.cuda(), pose_results_2d, dataset, with_track_id=False)

    with pytest.raises(NotImplementedError):
        _ = inference_pose_lifter_model(
            pose_model, pose_results_2d, dataset='test')
