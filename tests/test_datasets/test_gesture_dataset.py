# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch
from mmcv import Config
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS


def test_NVGesture_dataset():

    dataset = 'NVGestureDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/nvgesture.py').dataset_info

    dataset_class = DATASETS.get(dataset)

    data_cfg = dict(
        video_size=[320, 240],
        modality=['rgb', 'depth'],
        bbox_file='tests/data/nvgesture/bboxes.json',
    )

    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/nvgesture/test_nvgesture.lst',
        vid_prefix='tests/data/nvgesture/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/nvgesture/test_nvgesture.lst',
        vid_prefix='tests/data/nvgesture/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    assert custom_dataset.dataset_name == 'nvgesture'
    assert custom_dataset.test_mode is False
    assert len(custom_dataset) == 1
    sample = custom_dataset[0]

    # make pseudo prediction for evaluation
    sample['logits'] = {
        modal: torch.zeros(1, 25, 1)
        for modal in sample['modality']
    }
    sample['logits']['rgb'][:, sample['label']] = 1
    sample['logits']['depth'][:, (sample['label'] + 1) % 25] = 1
    sample['label'] = torch.tensor([sample['label']]).long()
    infos = custom_dataset.evaluate([sample], metric=['AP'])
    assert_almost_equal(infos['AP_rgb'], 1.0)
    assert_almost_equal(infos['AP_depth'], 0.0)
    assert_almost_equal(infos['AP_mean'], 0.5)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate([sample], metric='mAP')
