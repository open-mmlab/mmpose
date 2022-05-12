# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
from mmcv import Config
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS
from tests.utils.data_utils import convert_db_to_output


def test_animal_horse10_dataset():
    dataset = 'AnimalHorse10Dataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/horse10.py').dataset_info

    channel_cfg = dict(
        num_output_channels=22,
        dataset_joints=22,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 21
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 21
        ])

    data_cfg = dict(
        image_size=[256, 256],
        heatmap_size=[64, 64],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'])
    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/horse10/test_horse10.json',
        img_prefix='tests/data/horse10/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/horse10/test_horse10.json',
        img_prefix='tests/data/horse10/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.dataset_name == 'horse10'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 3
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric=['PCK'])
    assert_almost_equal(infos['PCK'], 1.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric='mAP')


def test_animal_fly_dataset():
    dataset = 'AnimalFlyDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/fly.py').dataset_info

    channel_cfg = dict(
        num_output_channels=32,
        dataset_joints=32,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ])

    data_cfg = dict(
        image_size=[192, 192],
        heatmap_size=[48, 48],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'])

    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/fly/test_fly.json',
        img_prefix='tests/data/fly/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/fly/test_fly.json',
        img_prefix='tests/data/fly/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.dataset_name == 'fly'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)

    infos = custom_dataset.evaluate(results, metric=['PCK'])
    assert_almost_equal(infos['PCK'], 1.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric='mAP')


def test_animal_locust_dataset():
    dataset = 'AnimalLocustDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/locust.py').dataset_info

    channel_cfg = dict(
        num_output_channels=35,
        dataset_joints=35,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                34
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
        ])

    data_cfg = dict(
        image_size=[160, 160],
        heatmap_size=[40, 40],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'])

    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/locust/test_locust.json',
        img_prefix='tests/data/locust/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/locust/test_locust.json',
        img_prefix='tests/data/locust/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.dataset_name == 'locust'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)

    infos = custom_dataset.evaluate(results, metric=['PCK'])
    assert_almost_equal(infos['PCK'], 1.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric='mAP')


def test_animal_zebra_dataset():
    dataset = 'AnimalZebraDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/zebra.py').dataset_info

    channel_cfg = dict(
        num_output_channels=9,
        dataset_joints=9,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    data_cfg = dict(
        image_size=[160, 160],
        heatmap_size=[40, 40],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'])

    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/zebra/test_zebra.json',
        img_prefix='tests/data/zebra/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/zebra/test_zebra.json',
        img_prefix='tests/data/zebra/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.dataset_name == 'zebra'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric=['PCK'])
    assert_almost_equal(infos['PCK'], 1.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric='mAP')


def test_animal_ATRW_dataset():
    dataset = 'AnimalATRWDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/atrw.py').dataset_info

    channel_cfg = dict(
        num_output_channels=15,
        dataset_joints=15,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    data_cfg = dict(
        image_size=[256, 256],
        heatmap_size=[64, 64],
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
        bbox_file='',
    )

    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/atrw/test_atrw.json',
        img_prefix='tests/data/atrw/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/atrw/test_atrw.json',
        img_prefix='tests/data/atrw/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.dataset_name == 'atrw'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric=['PCK'])

    # Test when gt annotations are absent
    del custom_dataset.coco.dataset['annotations']
    with pytest.warns(UserWarning):
        _ = custom_dataset.evaluate(results, metric='mAP')


def test_animal_Macaque_dataset():
    dataset = 'AnimalMacaqueDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/macaque.py').dataset_info

    channel_cfg = dict(
        num_output_channels=17,
        dataset_joints=17,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ])

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
        bbox_file='',
    )

    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/macaque/test_macaque.json',
        img_prefix='tests/data/macaque/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/macaque/test_macaque.json',
        img_prefix='tests/data/macaque/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.dataset_name == 'macaque'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric=['PCK'])

    # Test when gt annotations are absent
    del custom_dataset.coco.dataset['annotations']
    with pytest.warns(UserWarning):
        _ = custom_dataset.evaluate(results, metric='mAP')


def test_animalpose_dataset():
    dataset = 'AnimalPoseDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/animalpose.py').dataset_info

    channel_cfg = dict(
        num_output_channels=20,
        dataset_joints=20,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19
        ])

    data_cfg = dict(
        image_size=[256, 256],
        heatmap_size=[64, 64],
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
        bbox_file='',
    )

    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/animalpose/test_animalpose.json',
        img_prefix='tests/data/animalpose/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/animalpose/test_animalpose.json',
        img_prefix='tests/data/animalpose/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.dataset_name == 'animalpose'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric=['PCK'])

    # Test when gt annotations are absent
    del custom_dataset.coco.dataset['annotations']
    with pytest.warns(UserWarning):
        _ = custom_dataset.evaluate(results, metric='mAP')


def test_ap10k_dataset():
    dataset = 'AnimalAP10KDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/ap10k.py').dataset_info

    channel_cfg = dict(
        num_output_channels=17,
        dataset_joints=17,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ])

    data_cfg = dict(
        image_size=[256, 256],
        heatmap_size=[64, 64],
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
        bbox_file='',
    )

    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/ap10k/test_ap10k.json',
        img_prefix='tests/data/ap10k/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/ap10k/test_ap10k.json',
        img_prefix='tests/data/ap10k/',
        data_cfg=data_cfg_copy,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.dataset_name == 'ap10k'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)

    for output in results:
        # as there is only one box in each image for test
        output['bbox_ids'] = [0 for _ in range(len(output['bbox_ids']))]

    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(results, metric=['PCK'])

    # Test when gt annotations are absent
    del custom_dataset.coco.dataset['annotations']
    with pytest.warns(UserWarning):
        _ = custom_dataset.evaluate(results, metric='mAP')
