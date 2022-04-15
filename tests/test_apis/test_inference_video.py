# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from glob import glob

import mmcv

from mmpose.apis import (inference_top_down_video_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def test_top_down_video_multi_frame_demo():
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
    person_result.append({'bbox': [50, 50, 50, 100]})

    # test a viedo folder
    video_folder = 'tests/data/posetrack18/videos/000001_mpiinew_test'
    frames = sorted(glob(osp.join(
        video_folder, '*')))[:len(pose_model.cfg.data_cfg.frame_weight_test)]
    cur_frame = frames[0]

    # test the frames in the format of image paths
    pose_results, _ = inference_top_down_video_pose_model(
        pose_model,
        frames,
        person_result,
        format='xywh',
        dataset_info=dataset_info)
    # show the results
    vis_pose_result(
        pose_model, cur_frame, pose_results, dataset_info=dataset_info)

    # test a video file
    with tempfile.TemporaryDirectory() as tmpdir:
        # create video file from multiple frames
        video_path = osp.join(tmpdir, 'tmp_video.mp4')
        mmcv.frames2video(video_folder, video_path, fourcc='mp4v')
        video = mmcv.VideoReader(video_path)

        # get a sample for test
        cur_frame = video[0]
        frames = video[:len(pose_model.cfg.data_cfg.frame_weight_test)]

        person_result = []
        person_result.append({'bbox': [50, 75, 100, 150]})

        # test the frames in the format of image array
        pose_results, _ = inference_top_down_video_pose_model(
            pose_model,
            frames,
            person_result,
            format='xyxy',
            dataset_info=dataset_info)
        # show the results
        vis_pose_result(
            pose_model, cur_frame, pose_results, dataset_info=dataset_info)
