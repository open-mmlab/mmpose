# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from collections import defaultdict

import json_tricks as json
import numpy as np
from numpy.testing import assert_array_almost_equal

from mmpose.datasets.pipelines import (CenterSpatialCrop, CropValidClip,
                                       GestureRandomFlip, LoadVideoFromFile,
                                       ModalWiseChannelProcess,
                                       MultiFrameBBoxMerge,
                                       MultiModalVideoToTensor,
                                       RandomAlignedSpatialCrop,
                                       ResizedCropByBBox, ResizeGivenShortEdge,
                                       TemporalPooling, VideoNormalizeTensor)


def _check_flip(origin_vid, result_vid):
    """Check if the origin_video are flipped correctly."""
    l, h, w, c = origin_vid.shape

    for t in range(l):
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    if result_vid[t, i, j, k] != origin_vid[t, i, w - 1 - j,
                                                            k]:
                        return False
    return True


def _check_num_frames(video_results, num_frame):
    """Check if the video lengths match the given number of frames."""
    if video_results['num_frames'] != num_frame:
        return False
    if 'bbox' in video_results and len(video_results['bbox']) != num_frame:
        return False
    for video in video_results['video']:
        if video.shape[0] != num_frame:
            return False
    return True


def _check_size(video_results, size):
    """Check if the video sizes and size attributes match the given size."""
    for h in video_results['height']:
        if h != size[0]:
            return False
    for w in video_results['width']:
        if w != size[1]:
            return False
    for video in video_results['video']:
        if video.shape[1] != size[0]:
            return False
        if video.shape[2] != size[1]:
            return False
    return True


def _check_normalize(origin_video, result_video, norm_cfg):
    """Check if the origin_video are normalized correctly into result_video in
    a given norm_cfg."""
    target_video = result_video.clone()
    for i in range(3):
        target_video[i] *= norm_cfg['std'][i]
        target_video[i] += norm_cfg['mean'][i]
    assert_array_almost_equal(origin_video, target_video, decimal=4)


def test_gesture_pipeline():
    # test loading
    data_prefix = 'tests/data/nvgesture'

    results = defaultdict(list)
    results['modality'] = ['rgb', 'depth']
    results['label'] = 4
    with open(osp.join(data_prefix, 'bboxes.json'), 'r') as f:
        results['bbox'] = next(iter(json.load(f).values()))
    results['ann_info'] = dict(flip_pairs=((0, 1), (4, 5), (19, 20)))

    results['video_file'] = [
        osp.join(data_prefix, 'sk_color.avi'),
        osp.join(data_prefix, 'sk_depth.avi')
    ]
    transform = LoadVideoFromFile()
    results = transform(copy.deepcopy(results))

    assert results['video'][0].shape == (20, 240, 320, 3)
    assert results['video'][1].shape == (20, 240, 320, 3)

    # test CropValidClip
    results['valid_frames'] = ((2, 19), (1, 18))
    transform = CropValidClip()
    results_valid = transform(copy.deepcopy(results))
    assert _check_num_frames(results_valid, 17)
    assert (results_valid['video'][0] == results['video'][0][2:19]).all()
    assert (results_valid['video'][1] == results['video'][1][1:18]).all()

    # test TemporalPooling
    transform = TemporalPooling(ref_fps=15)
    results_temp_pool = transform(copy.deepcopy(results_valid))
    assert _check_num_frames(results_temp_pool, 9)

    transform = TemporalPooling(length=10)
    results_temp_pool = transform(copy.deepcopy(results_valid))
    assert _check_num_frames(results_temp_pool, 10)
    del results_temp_pool

    # test ResizeGivenShortEdge
    transform = ResizeGivenShortEdge(length=256)
    results_resize = transform(copy.deepcopy(results_valid))
    assert _check_size(results_resize, (256, 341))
    del results_resize

    # test MultiFrameBBoxMerge
    transform = MultiFrameBBoxMerge()
    results_bbox_merge = transform(copy.deepcopy(results_valid))
    target_bbox = np.array([96.07688, 92.26083, 316.5224,
                            231.98422]).astype(np.float32)
    assert_array_almost_equal(results_bbox_merge['bbox'], target_bbox, 4)

    # test ResizedCropByBBox
    transform = ResizedCropByBBox(
        size=112, scale=(0.8, 1.2), ratio=(0.8, 1.2), shift=0.3)
    results_resize_crop = transform(copy.deepcopy(results_bbox_merge))
    assert _check_size(results_resize_crop, (112, 112))
    del results_bbox_merge

    # test GestureRandomFlip
    transform = GestureRandomFlip(prob=1.0)
    results_flip = transform(copy.deepcopy(results_resize_crop))
    assert results_flip['label'] == 5
    assert _check_size(results_flip, (112, 112))
    assert _check_flip(results_flip['video'][0],
                       results_resize_crop['video'][0])
    assert _check_flip(results_flip['video'][1],
                       results_resize_crop['video'][1])
    del results_resize_crop

    # test RandomAlignedSpatialCrop & CenterSpatialCrop
    transform = RandomAlignedSpatialCrop(length=112)
    results_crop = transform(copy.deepcopy(results_valid))
    assert _check_size(results_crop, (112, 112))

    transform = CenterSpatialCrop(length=112)
    results_crop = transform(copy.deepcopy(results_valid))
    assert _check_size(results_crop, (112, 112))
    del results_crop

    # test ModalWiseChannelProcess
    transform = ModalWiseChannelProcess()
    results_modal_proc = transform(copy.deepcopy(results_valid))
    for i, modal in enumerate(results_modal_proc['modality']):
        if modal == 'rgb':
            assert_array_almost_equal(
                results_modal_proc['video'][i][..., ::-1],
                results_valid['video'][i])
        if modal == 'depth':
            assert_array_almost_equal(results_modal_proc['video'][i],
                                      results_valid['video'][i][..., :1])
    del results_valid

    # test MultiModalVideoToTensor
    transform = MultiModalVideoToTensor()
    results_tensor = transform(copy.deepcopy(results_modal_proc))
    for i, video in enumerate(results_tensor['video']):
        assert video.max() <= 1.0 and video.min() >= 0.0
        assert video.shape[1:] == results_modal_proc['video'][i].shape[:-1]
        assert video.shape[0] == results_modal_proc['video'][i].shape[-1]
    del results_modal_proc

    # test VideoNormalizeTensor
    norm_cfg = {}
    norm_cfg['mean'] = [0.485, 0.456, 0.406]
    norm_cfg['std'] = [0.229, 0.224, 0.225]
    transform = VideoNormalizeTensor(**norm_cfg)
    results_norm = transform(copy.deepcopy(results_tensor))
    _check_normalize(results_tensor['video'][0], results_norm['video'][0],
                     norm_cfg)
