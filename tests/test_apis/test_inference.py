# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from glob import glob

import json_tricks as json
import mmcv
import numpy as np

from mmpose.apis import (collect_multi_frames, inference_bottom_up_pose_model,
                         inference_gesture_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo


def test_top_down_demo():
    # COCO demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'coco/res50_coco_256x192.py',
        None,
        device='cpu')
    image_name = 'tests/data/coco/000000000785.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))

    person_result = []
    person_result.append({'bbox': [50, 50, 50, 100]})
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # MPII demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/deeppose/'
        'mpii/res50_mpii_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/mpii/004645041.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))
    person_result = []
    person_result.append({'bbox': [50, 50, 50, 100]})
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # AIC demo
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'aic/res50_aic_256x192.py',
        None,
        device='cpu')
    image_name = 'tests/data/aic/054d9ce9201beffc76e5ff2169d2af2f027002ca.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # OneHand10K demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'onehand10k/res50_onehand10k_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/onehand10k/9.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # InterHand2DDataset demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'interhand2d/res50_interhand2d_all_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/interhand2.6m/image2017.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # Face300WDataset demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/'
        '300w/res50_300w_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/300w/indoor_020.png'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # FaceAFLWDataset demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'aflw/res50_aflw_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/aflw/image04476.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # FaceCOFWDataset demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'cofw/res50_cofw_256x256.py',
        None,
        device='cpu')
    image_name = 'tests/data/cofw/001766.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))
    # test a single image, with a list of bboxes.
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # test video pose demo
    # PoseWarper + PoseTrack18 demo
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/'
        'hrnet_w48_posetrack18_384x288_posewarper_stage2.py',
        None,
        device='cpu')
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))

    person_result = []
    # the last value is the confidence of bbox
    person_result.append({'bbox': [50, 50, 50, 100, 0.5]})

    # test a viedo folder
    num_frames = len(pose_model.cfg.data_cfg.frame_weight_test)
    video_folder = 'tests/data/posetrack18/videos/000001_mpiinew_test'
    frames = sorted(glob(osp.join(video_folder, '*.jpg')))[:num_frames]
    cur_frame = frames[0]

    # test the frames in the format of image paths
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        frames,
        person_result,
        format='xywh',
        bbox_thr=0.3,
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, cur_frame, pose_results, dataset_info=dataset_info)

    # test when the person_result is None
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        frames,
        person_results=None,
        format='xywh',
        dataset_info=dataset_info)

    # test a video file
    video_path = 'tests/data/posetrack18/videos/000001_mpiinew_test/'\
        '000001_mpiinew_test.mp4'
    video = mmcv.VideoReader(video_path)

    # get a sample for test
    cur_frame = video[0]
    frames = video[:num_frames]

    person_result = []
    person_result.append({'bbox': [50, 75, 100, 150, 0.6]})

    # test the frames in the format of image array
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        frames,
        person_result,
        bbox_thr=0.9,
        format='xyxy',
        dataset_info=dataset_info)

    # # test the frames in the format of image array
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        frames,
        person_results=None,
        format='xyxy',
        dataset_info=dataset_info)


def test_bottom_up_demo():

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/body/2d_kpt_sview_rgb_img/associative_embedding/'
        'coco/res50_coco_512x512.py',
        None,
        device='cpu')

    image_name = 'tests/data/coco/000000000785.jpg'
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))

    pose_results, _ = inference_bottom_up_pose_model(
        pose_model, image_name, dataset_info=dataset_info)

    # show the results
    vis_pose_result(
        pose_model, image_name, pose_results, dataset_info=dataset_info)

    # test dataset_info without sigmas
    pose_model_copy = copy.deepcopy(pose_model)

    pose_model_copy.cfg.data.test.dataset_info.pop('sigmas')
    pose_results, _ = inference_bottom_up_pose_model(
        pose_model_copy, image_name, dataset_info=dataset_info)


def test_process_mmdet_results():
    det_results = [np.array([0, 0, 100, 100])]
    det_mask_results = None

    _ = process_mmdet_results(
        mmdet_results=(det_results, det_mask_results), cat_id=1)

    _ = process_mmdet_results(mmdet_results=det_results, cat_id=1)


def test_collect_multi_frames():
    # video file for test
    video_path = 'tests/data/posetrack18/videos/000001_mpiinew_test/'\
        '000001_mpiinew_test.mp4'
    video = mmcv.VideoReader(video_path)

    frame_id = 0
    indices = [-1, -2, 0, 1, 2]

    _ = collect_multi_frames(video, frame_id, indices, online=True)

    _ = collect_multi_frames(video, frame_id, indices, online=False)


def test_hand_gesture_demo():

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        'configs/hand/gesture_sview_rgbd_vid/mtut/nvgesture/'
        'i3d_nvgesture_bbox_112x112_fps15.py',
        None,
        device='cpu')

    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    video_files = [
        'tests/data/nvgesture/sk_color.avi',
        'tests/data/nvgesture/sk_depth.avi'
    ]
    with open('tests/data/nvgesture/bboxes.json', 'r') as f:
        bbox = next(iter(json.load(f).values()))

    pred_label, _ = inference_gesture_model(pose_model, video_files, bbox,
                                            dataset_info)
