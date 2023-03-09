# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmpose.core import (aggregate_scale, aggregate_stage_flip,
                         flip_feature_maps, get_group_preds, split_ae_outputs)


def test_split_ae_outputs():
    fake_outputs = [torch.zeros((1, 4, 2, 2))]
    heatmaps, tags = split_ae_outputs(
        fake_outputs,
        num_joints=4,
        with_heatmaps=[False],
        with_ae=[True],
        select_output_index=[0])


def test_flip_feature_maps():
    fake_outputs = [torch.zeros((1, 4, 2, 2))]
    _ = flip_feature_maps(fake_outputs, None)
    _ = flip_feature_maps(fake_outputs, flip_index=[1, 0])


def test_aggregate_stage_flip():
    fake_outputs = [torch.zeros((1, 4, 2, 2))]
    fake_flip_outputs = [torch.ones((1, 4, 2, 2))]
    output = aggregate_stage_flip(
        fake_outputs,
        fake_flip_outputs,
        index=-1,
        project2image=True,
        size_projected=(4, 4),
        align_corners=False,
        aggregate_stage='concat',
        aggregate_flip='average')
    assert isinstance(output, list)

    output = aggregate_stage_flip(
        fake_outputs,
        fake_flip_outputs,
        index=-1,
        project2image=True,
        size_projected=(4, 4),
        align_corners=False,
        aggregate_stage='average',
        aggregate_flip='average')
    assert isinstance(output, list)

    output = aggregate_stage_flip(
        fake_outputs,
        fake_flip_outputs,
        index=-1,
        project2image=True,
        size_projected=(4, 4),
        align_corners=False,
        aggregate_stage='average',
        aggregate_flip='concat')
    assert isinstance(output, list)

    output = aggregate_stage_flip(
        fake_outputs,
        fake_flip_outputs,
        index=-1,
        project2image=True,
        size_projected=(4, 4),
        align_corners=False,
        aggregate_stage='concat',
        aggregate_flip='concat')
    assert isinstance(output, list)


def test_aggregate_scale():
    fake_outputs = [torch.zeros((1, 4, 2, 2)), torch.zeros((1, 4, 2, 2))]
    output = aggregate_scale(
        fake_outputs, align_corners=False, aggregate_scale='average')
    assert isinstance(output, torch.Tensor)
    assert output.shape == fake_outputs[0].shape

    output = aggregate_scale(
        fake_outputs, size_projected=(4, 3), aggregate_scale='average')
    assert isinstance(output, torch.Tensor)
    assert output.shape[:2] == fake_outputs[0].shape[:2]
    assert output.shape[2:] == (3, 4)

    output = aggregate_scale(
        fake_outputs, align_corners=False, aggregate_scale='unsqueeze_concat')

    assert isinstance(output, torch.Tensor)
    assert len(output.shape) == len(fake_outputs[0].shape) + 1


def test_get_group_preds():
    fake_grouped_joints = [np.array([[[0, 0], [1, 1]]])]
    results = get_group_preds(
        fake_grouped_joints,
        center=np.array([0, 0]),
        scale=np.array([1, 1]),
        heatmap_size=np.array([2, 2]))
    assert not results == []

    results = get_group_preds(
        fake_grouped_joints,
        center=np.array([0, 0]),
        scale=np.array([1, 1]),
        heatmap_size=np.array([2, 2]),
        use_udp=True)
    assert not results == []
