import pytest

from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


def test_top_down_demo():
    # COCO demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'coco/res50_coco_256x192.py',
        None,
        device='cpu')
    image_name = 'tests/data/coco/000000000785.jpg'

    person_result = []
    person_result.append({'bbox': [50, 50, 50, 100]})
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model, image_name, person_result, format='xywh')
    # show the results
    vis_pose_result(pose_model, image_name, pose_results)

    # AIC demo
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'aic/res50_aic_256x192.py',
        None,
        device='cpu')
    image_name = 'tests/data/aic/054d9ce9201beffc76e5ff2169d2af2f027002ca.jpg'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset='TopDownAicDataset')
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='TopDownAicDataset')

    # OneHand10K demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'onehand10k/res50_onehand10k_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/onehand10k/9.jpg'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset='OneHand10KDataset')
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='OneHand10KDataset')

    # InterHand2DDataset demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'interhand2d/res50_interhand2d_all_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/interhand2.6m/image2017.jpg'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset='InterHand2DDataset')
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='InterHand2DDataset')

    # Face300WDataset demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/'
        '300w/res50_300w_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/300w/indoor_020.png'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset='Face300WDataset')
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='Face300WDataset')

    # FaceAFLWDataset demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'aflw/res50_aflw_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/aflw/image04476.jpg'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset='FaceAFLWDataset')
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='FaceAFLWDataset')

    # FaceCOFWDataset demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'cofw/res50_cofw_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/cofw/001766.jpg'
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset='FaceCOFWDataset')
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='FaceCOFWDataset')

    with pytest.raises(NotImplementedError):
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_result,
            format='xywh',
            dataset='test')


def test_bottom_up_demo():

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/associative_embedding/'
        'coco/res50_coco_512x512.py',
        None,
        device='cpu')

    image_name = 'tests/data/coco/000000000785.jpg'

    pose_results, _ = inference_bottom_up_pose_model(pose_model, image_name)

    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset='BottomUpCocoDataset')
