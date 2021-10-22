# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmpose.apis import (extract_pose_sequence, get_track_id,
                         inference_bottom_up_pose_model,
                         inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model,
                         vis_3d_pose_result, vis_pose_result,
                         vis_pose_tracking_result)


def test_inference_without_dataset_info():
    # Top down
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'coco/res50_coco_256x192.py',
        None,
        device='cpu')

    if 'dataset_info' in pose_model.cfg:
        _ = pose_model.cfg.pop('dataset_info')

    image_name = 'tests/data/coco/000000000785.jpg'
    person_result = []
    person_result.append({'bbox': [50, 50, 50, 100]})

    with pytest.warns(DeprecationWarning):
        pose_results, _ = inference_top_down_pose_model(
            pose_model, image_name, person_result, format='xywh')

    with pytest.warns(DeprecationWarning):
        vis_pose_result(pose_model, image_name, pose_results)

    with pytest.raises(NotImplementedError):
        with pytest.warns(DeprecationWarning):
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                image_name,
                person_result,
                format='xywh',
                dataset='test')

    # Bottom up
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/associative_embedding/'
        'coco/res50_coco_512x512.py',
        None,
        device='cpu')
    if 'dataset_info' in pose_model.cfg:
        _ = pose_model.cfg.pop('dataset_info')

    image_name = 'tests/data/coco/000000000785.jpg'

    with pytest.warns(DeprecationWarning):
        pose_results, _ = inference_bottom_up_pose_model(
            pose_model, image_name)
    with pytest.warns(DeprecationWarning):
        vis_pose_result(pose_model, image_name, pose_results)

    # Top down tracking
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'coco/res50_coco_256x192.py',
        None,
        device='cpu')

    if 'dataset_info' in pose_model.cfg:
        _ = pose_model.cfg.pop('dataset_info')

    image_name = 'tests/data/coco/000000000785.jpg'
    person_result = [{'bbox': [50, 50, 50, 100]}]

    with pytest.warns(DeprecationWarning):
        pose_results, _ = inference_top_down_pose_model(
            pose_model, image_name, person_result, format='xywh')

    pose_results, _ = get_track_id(pose_results, [], next_id=0)

    with pytest.warns(DeprecationWarning):
        vis_pose_tracking_result(pose_model, image_name, pose_results)

    with pytest.raises(NotImplementedError):
        with pytest.warns(DeprecationWarning):
            vis_pose_tracking_result(
                pose_model, image_name, pose_results, dataset='test')

    # Bottom up tracking
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/associative_embedding/'
        'coco/res50_coco_512x512.py',
        None,
        device='cpu')

    if 'dataset_info' in pose_model.cfg:
        _ = pose_model.cfg.pop('dataset_info')

    image_name = 'tests/data/coco/000000000785.jpg'
    with pytest.warns(DeprecationWarning):
        pose_results, _ = inference_bottom_up_pose_model(
            pose_model, image_name)

    pose_results, next_id = get_track_id(pose_results, [], next_id=0)

    with pytest.warns(DeprecationWarning):
        vis_pose_tracking_result(
            pose_model,
            image_name,
            pose_results,
            dataset='BottomUpCocoDataset')

    # Pose lifting
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

    if 'dataset_info' in pose_model.cfg:
        _ = pose_model.cfg.pop('dataset_info')

    pose_results_2d = [[pose_det_result]]

    dataset = pose_model.cfg.data['test']['type']

    pose_results_2d = extract_pose_sequence(
        pose_results_2d, frame_idx=0, causal=False, seq_len=1, step=1)

    with pytest.warns(DeprecationWarning):
        _ = inference_pose_lifter_model(
            pose_model, pose_results_2d, dataset, with_track_id=False)

    with pytest.warns(DeprecationWarning):
        pose_lift_results = inference_pose_lifter_model(
            pose_model, pose_results_2d, dataset, with_track_id=True)

    for res in pose_lift_results:
        res['title'] = 'title'
    with pytest.warns(DeprecationWarning):
        vis_3d_pose_result(
            pose_model,
            pose_lift_results,
            img=pose_results_2d[0][0]['image_name'],
            dataset=dataset)

    with pytest.raises(NotImplementedError):
        with pytest.warns(DeprecationWarning):
            _ = inference_pose_lifter_model(
                pose_model, pose_results_2d, dataset='test')
