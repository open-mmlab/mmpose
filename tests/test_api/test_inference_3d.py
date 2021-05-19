import numpy as np

from mmpose.apis import (inference_pose_lifter_model, init_pose_model,
                         vis_3d_pose_result)


def test_pose_lifter_demo():
    # H36M demo
    pose_model = init_pose_model(
        'configs/body3d/simple_baseline/h36m/simple3Dbaseline_h36m.py',
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
