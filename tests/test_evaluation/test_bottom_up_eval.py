import copy

import numpy as np
import torch

from mmpose.core import (aggregate_results, get_group_preds,
                         get_multi_stage_outputs)


def test_get_multi_stage_outputs():
    fake_outputs = [torch.zeros((1, 4, 2, 2))]
    fake_flip_outputs = [torch.ones((1, 4, 2, 2))]
    # outputs_flip
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                outputs_flip=None,
                                num_joints=4, with_heatmaps=[False],
                                with_ae=[True])
    assert heatmaps == []
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                outputs_flip=None,
                                num_joints=2, with_heatmaps=[True],
                                with_ae=[True])
    assert len(heatmaps) == 1
    flip_index = [1, 0]
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                outputs_flip=fake_flip_outputs,
                                num_joints=2, with_heatmaps=[True],
                                with_ae=[True], flip_index=flip_index)
    assert len(heatmaps) == 2
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                tag_per_joint=False,
                                outputs_flip=fake_flip_outputs,
                                num_joints=2, with_heatmaps=[True],
                                with_ae=[True], flip_index=flip_index)
    assert len(heatmaps) == 2
    # with heatmaps & with ae
    fake_outputs = [torch.zeros((1, 4, 2, 2)), torch.ones((1, 2, 4, 4))]
    fake_flip_outputs = [torch.ones((1, 4, 2, 2)), torch.ones((1, 2, 4, 4))]
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                outputs_flip=None,
                                num_joints=2, with_heatmaps=[True, False],
                                with_ae=[True, True])
    assert torch.allclose(heatmaps[0], torch.tensor(0.))
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                outputs_flip=fake_flip_outputs,
                                num_joints=2, with_heatmaps=[True, True],
                                with_ae=[True, False])
    assert torch.allclose(heatmaps[0], torch.tensor(0.5))
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                outputs_flip=fake_flip_outputs,
                                num_joints=2, with_heatmaps=[True, False],
                                with_ae=[True, False], flip_index=flip_index)
    assert torch.allclose(heatmaps[0], torch.tensor(0.))
    # size_projected
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                outputs_flip=None,
                                num_joints=2, with_heatmaps=[True, True],
                                with_ae=[True, False],
                                size_projected=(8, 8))
    assert heatmaps[0].shape == torch.Size([1, 2, 8, 8])
    outputs, heatmaps, tags = \
        get_multi_stage_outputs(outputs=copy.deepcopy(fake_outputs),
                                outputs_flip=fake_flip_outputs,
                                num_joints=2, with_heatmaps=[True, True],
                                with_ae=[True, False],
                                align_corners=True)
    assert torch.allclose(heatmaps[0], torch.tensor(0.5))


def test_aggregate_results():
    fake_heatmaps = [torch.zeros((1, 2, 2, 2))]
    fake_tags = [torch.zeros((1, 2, 2, 2))]
    aggregated_heatmaps, tags_list = \
        aggregate_results(scale=1, aggregated_heatmaps=None, tags_list=[],
                          heatmaps=fake_heatmaps, tags=fake_tags,
                          test_scale_factor=[1], project2image=True,
                          flip_test=False)
    assert torch.allclose(aggregated_heatmaps, torch.tensor(0.))
    fake_aggr_heatmaps = torch.ones(1, 2, 2, 2)
    aggregated_heatmaps, tags_list = \
        aggregate_results(scale=1, aggregated_heatmaps=fake_aggr_heatmaps,
                          tags_list=[], heatmaps=fake_heatmaps,
                          tags=fake_tags, test_scale_factor=[1],
                          project2image=True, flip_test=False)
    assert torch.allclose(aggregated_heatmaps, torch.tensor(1.))
    aggregated_heatmaps, tags_list = \
        aggregate_results(scale=1, aggregated_heatmaps=fake_aggr_heatmaps,
                          tags_list=[], heatmaps=fake_heatmaps,
                          tags=fake_tags, test_scale_factor=[1],
                          project2image=True, flip_test=False,
                          align_corners=True)
    assert torch.allclose(aggregated_heatmaps, torch.tensor(1.))
    fake_heatmaps = [torch.zeros((1, 2, 2, 2)), torch.ones((1, 2, 2, 2))]
    fake_aggr_heatmaps = torch.ones(1, 2, 4, 4)
    aggregated_heatmaps, tags_list = \
        aggregate_results(scale=1, aggregated_heatmaps=fake_aggr_heatmaps,
                          tags_list=[], heatmaps=fake_heatmaps,
                          tags=fake_tags, test_scale_factor=[1],
                          project2image=False, flip_test=True)
    assert aggregated_heatmaps.shape == torch.Size((1, 2, 4, 4))
    aggregated_heatmaps, tags_list = \
        aggregate_results(scale=2, aggregated_heatmaps=fake_aggr_heatmaps,
                          tags_list=[], heatmaps=fake_heatmaps,
                          tags=fake_tags, test_scale_factor=[1, 2],
                          project2image=False, flip_test=True)
    assert aggregated_heatmaps.shape == torch.Size((1, 2, 4, 4))


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
