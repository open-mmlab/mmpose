# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import pytest
from mmcv import Config
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS
from tests.utils.data_utils import convert_db_to_output


def test_deepfashion_dataset():
    dataset = 'DeepFashionDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/deepfashion_full.py').dataset_info
    # test JHMDB datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=8,
        dataset_joints=8,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7])

    data_cfg = dict(
        image_size=[192, 256],
        heatmap_size=[48, 64],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        soft_nms=False,
        nms_thr=1.0,
        oks_thr=0.9,
        vis_thr=0.2,
        use_gt_bbox=True,
        det_bbox_thr=0.0,
        image_thr=0.0,
        bbox_file='')

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/fld/test_fld.json',
        img_prefix='tests/data/fld/',
        subset='full',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'deepfashion_full'

    image_id = 128
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric=['PCK', 'EPE', 'AUC'])
    assert_almost_equal(infos['PCK'], 1.0)
    assert_almost_equal(infos['AUC'], 0.95)
    assert_almost_equal(infos['EPE'], 0.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric='mAP')
