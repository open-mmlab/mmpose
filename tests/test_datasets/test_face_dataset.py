# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest.mock import MagicMock

import pytest
from mmcv import Config
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS
from tests.utils.data_utils import convert_db_to_output


def test_face_300W_dataset():
    dataset = 'Face300WDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/300w.py').dataset_info
    # test Face 300W datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=68,
        dataset_joints=68,
        dataset_channel=[
            list(range(68)),
        ],
        inference_channel=list(range(68)))

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
        ann_file='tests/data/300w/test_300w.json',
        img_prefix='tests/data/300w/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/300w/test_300w.json',
        img_prefix='tests/data/300w/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    assert custom_dataset.dataset_name == '300w'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric=['NME'])
    assert_almost_equal(infos['NME'], 0.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='mAP')


def test_face_coco_wholebody_dataset():
    dataset = 'FaceCocoWholeBodyDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/coco_wholebody_face.py').dataset_info
    # test Face wholebody datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=68,
        dataset_joints=68,
        dataset_channel=[
            list(range(68)),
        ],
        inference_channel=list(range(68)))

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
        ann_file='tests/data/coco/test_coco_wholebody.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/coco/test_coco_wholebody.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 4
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric=['NME'])
    assert_almost_equal(infos['NME'], 0.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='mAP')


def test_face_AFLW_dataset():
    dataset = 'FaceAFLWDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/aflw.py').dataset_info
    # test Face AFLW datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=19,
        dataset_joints=19,
        dataset_channel=[
            list(range(19)),
        ],
        inference_channel=list(range(19)))

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
        ann_file='tests/data/aflw/test_aflw.json',
        img_prefix='tests/data/aflw/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/aflw/test_aflw.json',
        img_prefix='tests/data/aflw/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    assert custom_dataset.dataset_name == 'aflw'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric=['NME'])
    assert_almost_equal(infos['NME'], 0.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='mAP')


def test_face_WFLW_dataset():
    dataset = 'FaceWFLWDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/wflw.py').dataset_info
    # test Face WFLW datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=98,
        dataset_joints=98,
        dataset_channel=[
            list(range(98)),
        ],
        inference_channel=list(range(98)))

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
        ann_file='tests/data/wflw/test_wflw.json',
        img_prefix='tests/data/wflw/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/wflw/test_wflw.json',
        img_prefix='tests/data/wflw/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    assert custom_dataset.dataset_name == 'wflw'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric=['NME'])
    assert_almost_equal(infos['NME'], 0.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='mAP')


def test_face_COFW_dataset():
    dataset = 'FaceCOFWDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/cofw.py').dataset_info
    # test Face COFW datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=29,
        dataset_joints=29,
        dataset_channel=[
            list(range(29)),
        ],
        inference_channel=list(range(29)))

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
        ann_file='tests/data/cofw/test_cofw.json',
        img_prefix='tests/data/cofw/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/cofw/test_cofw.json',
        img_prefix='tests/data/cofw/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    assert custom_dataset.dataset_name == 'cofw'
    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    results = convert_db_to_output(custom_dataset.db)
    infos = custom_dataset.evaluate(results, metric=['NME'])
    assert_almost_equal(infos['NME'], 0.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='mAP')
