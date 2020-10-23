import pytest

from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


def test_top_down_demo():
    # COCO demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/top_down/resnet/coco/res50_coco_256x192.py',
        None,
        device='cpu')
    image_name = 'tests/data/coco/000000000785.jpg'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model, image_name, [[50, 50, 50, 100]], format='xywh')
    # show the results
    vis_pose_result(pose_model, image_name, pose_results)

    # AIC demo
    pose_model = init_pose_model(
        'configs/top_down/resnet/aic/res50_aic_256x192.py', None, device='cpu')
    image_name = 'tests/data/aic/054d9ce9201beffc76e5ff2169d2af2f027002ca.jpg'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name, [[50, 50, 50, 100]],
        format='xywh',
        dataset='TopDownAicDataset')
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='TopDownAicDataset')

    # OneHand10K demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/hand/resnet/onehand10k/res50_onehand10k_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/onehand10k/9.jpg'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name, [[50, 50, 50, 100]],
        format='xywh',
        dataset='OneHand10KDataset')
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='OneHand10KDataset')

    with pytest.raises(NotImplementedError):
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            image_name, [[50, 50, 50, 100]],
            format='xywh',
            dataset='test')


def test_bottom_up_demo():

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/bottom_up/resnet/coco/res50_coco_512x512.py',
        None,
        device='cpu')

    image_name = 'tests/data/coco/000000000785.jpg'

    pose_results, _ = inference_bottom_up_pose_model(pose_model, image_name)

    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='BottomUpCocoDataset')
