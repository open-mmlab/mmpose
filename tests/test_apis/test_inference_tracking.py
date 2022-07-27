# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmpose.apis import (get_track_id, inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         vis_pose_tracking_result)
from mmpose.datasets.dataset_info import DatasetInfo


def test_top_down_pose_tracking_demo():
    # COCO demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'coco/res50_coco_256x192.py',
        None,
        device='cpu')
    image_name = 'tests/data/coco/000000000785.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test']['dataset_info'])
    person_result = [{'bbox': [50, 50, 50, 100]}]

    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    pose_results, next_id = get_track_id(pose_results, [], next_id=0)
    # show the results
    vis_pose_tracking_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)
    pose_results_last = pose_results

    # AIC demo
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'aic/res50_aic_256x192.py',
        None,
        device='cpu')
    image_name = 'tests/data/aic/054d9ce9201beffc76e5ff2169d2af2f027002ca.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test']['dataset_info'])
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    pose_results, next_id = get_track_id(pose_results, pose_results_last,
                                         next_id)
    for pose_result in pose_results:
        del pose_result['bbox']
    pose_results, next_id = get_track_id(pose_results, pose_results_last,
                                         next_id)

    # show the results
    vis_pose_tracking_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # OneHand10K demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'onehand10k/res50_onehand10k_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/onehand10k/9.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test']['dataset_info'])
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name, [{
            'bbox': [10, 10, 30, 30]
        }],
        format='xywh',
        dataset_info=dataset_info)
    pose_results, next_id = get_track_id(pose_results, pose_results_last,
                                         next_id)
    # show the results
    vis_pose_tracking_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # InterHand2D demo
    pose_model = init_pose_model(
        'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'interhand2d/res50_interhand2d_all_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/interhand2.6m/image2017.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test']['dataset_info'])
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name, [{
            'bbox': [50, 50, 0, 0]
        }],
        format='xywh',
        dataset_info=dataset_info)
    pose_results, next_id = get_track_id(pose_results, [], next_id=0)
    # show the results
    vis_pose_tracking_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)
    pose_results_last = pose_results

    # MPII demo
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'mpii/res50_mpii_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/mpii/004645041.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test']['dataset_info'])
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name, [{
            'bbox': [50, 50, 0, 0]
        }],
        format='xywh',
        dataset_info=dataset_info)
    pose_results, next_id = get_track_id(pose_results, pose_results_last,
                                         next_id)
    # show the results
    vis_pose_tracking_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)


def test_bottom_up_pose_tracking_demo():
    # COCO demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/associative_embedding/'
        'coco/res50_coco_512x512.py',
        None,
        device='cpu')

    image_name = 'tests/data/coco/000000000785.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test']['dataset_info'])

    pose_results, _ = inference_bottom_up_pose_model(
        pose_model, image_name, dataset_info=dataset_info)

    pose_results, next_id = get_track_id(pose_results, [], next_id=0)

    # show the results
    vis_pose_tracking_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    pose_results_last = pose_results

    # oks
    pose_results, next_id = get_track_id(
        pose_results,
        pose_results_last,
        next_id=next_id,
        use_oks=True,
        sigmas=getattr(dataset_info, 'sigmas', None))

    pose_results_last = pose_results

    # one_euro (will be deprecated)
    with pytest.deprecated_call():
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id=next_id,
            use_one_euro=True)
