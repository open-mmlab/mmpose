# Copyright (c) OpenMMLab. All rights reserved.
import copy
import tempfile

import pytest
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS
from tests.utils.data_utils import convert_db_to_output


def test_top_down_OneHand10K_dataset_compatibility():
    dataset = 'OneHand10KDataset'
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=21,
        dataset_joints=21,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
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
    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/onehand10k/test_onehand10k.json',
            img_prefix='tests/data/onehand10k/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=True)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/onehand10k/test_onehand10k.json',
            img_prefix='tests/data/onehand10k/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 4
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK', 'EPE', 'AUC'])
        assert_almost_equal(infos['PCK'], 1.0)
        assert_almost_equal(infos['AUC'], 0.95)
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_FreiHand_dataset_compatibility():
    dataset = 'FreiHandDataset'
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=21,
        dataset_joints=21,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ])

    data_cfg = dict(
        image_size=[224, 224],
        heatmap_size=[56, 56],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'])
    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/freihand/test_freihand.json',
            img_prefix='tests/data/freihand/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=True)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/freihand/test_freihand.json',
            img_prefix='tests/data/freihand/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 8
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK', 'EPE', 'AUC'])
        assert_almost_equal(infos['PCK'], 1.0)
        assert_almost_equal(infos['AUC'], 0.95)
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_RHD_dataset_compatibility():
    dataset = 'Rhd2DDataset'
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=21,
        dataset_joints=21,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
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
    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/rhd/test_rhd.json',
            img_prefix='tests/data/rhd/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=True)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/rhd/test_rhd.json',
            img_prefix='tests/data/rhd/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 3
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK', 'EPE', 'AUC'])
        assert_almost_equal(infos['PCK'], 1.0)
        assert_almost_equal(infos['AUC'], 0.95)
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_Panoptic_dataset_compatibility():
    dataset = 'PanopticDataset'
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=21,
        dataset_joints=21,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
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
    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/panoptic/test_panoptic.json',
            img_prefix='tests/data/panoptic/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=True)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/panoptic/test_panoptic.json',
            img_prefix='tests/data/panoptic/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 4
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir,
                                        ['PCKh', 'EPE', 'AUC'])
        assert_almost_equal(infos['PCKh'], 1.0)
        assert_almost_equal(infos['AUC'], 0.95)
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_InterHand2D_dataset_compatibility():
    dataset = 'InterHand2DDataset'
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=21,
        dataset_joints=21,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
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
    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/interhand2.6m/test_interhand2.6m_data.json',
            camera_file='tests/data/interhand2.6m/'
            'test_interhand2.6m_camera.json',
            joint_file='tests/data/interhand2.6m/'
            'test_interhand2.6m_joint_3d.json',
            img_prefix='tests/data/interhand2.6m/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=True)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/interhand2.6m/test_interhand2.6m_data.json',
            camera_file='tests/data/interhand2.6m/'
            'test_interhand2.6m_camera.json',
            joint_file='tests/data/interhand2.6m/'
            'test_interhand2.6m_joint_3d.json',
            img_prefix='tests/data/interhand2.6m/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 4
    assert len(custom_dataset.db) == 6

    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK', 'EPE', 'AUC'])
        print(infos, flush=True)
        assert_almost_equal(infos['PCK'], 1.0)
        assert_almost_equal(infos['AUC'], 0.95)
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_InterHand3D_dataset_compatibility():
    dataset = 'InterHand3DDataset'
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=42,
        dataset_joints=42,
        dataset_channel=[
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                34, 35, 36, 37, 38, 39, 40, 41
            ],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41
        ])

    data_cfg = dict(
        image_size=[256, 256],
        heatmap_size=[64, 64, 64],
        heatmap3d_depth_bound=400.0,
        heatmap_size_root=64,
        root_depth_bound=400.0,
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'])
    # Test
    data_cfg_copy = copy.deepcopy(data_cfg)
    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/interhand2.6m/test_interhand2.6m_data.json',
            camera_file='tests/data/interhand2.6m/'
            'test_interhand2.6m_camera.json',
            joint_file='tests/data/interhand2.6m/'
            'test_interhand2.6m_joint_3d.json',
            img_prefix='tests/data/interhand2.6m/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=True)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/interhand2.6m/test_interhand2.6m_data.json',
            camera_file='tests/data/interhand2.6m/'
            'test_interhand2.6m_camera.json',
            joint_file='tests/data/interhand2.6m/'
            'test_interhand2.6m_joint_3d.json',
            img_prefix='tests/data/interhand2.6m/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 4
    assert len(custom_dataset.db) == 4

    _ = custom_dataset[0]

    outputs = convert_db_to_output(
        custom_dataset.db, keys=['rel_root_depth', 'hand_type'], is_3d=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir,
                                        ['MRRPE', 'MPJPE', 'Handedness_acc'])
        assert_almost_equal(infos['MRRPE'], 0.0, decimal=5)
        assert_almost_equal(infos['MPJPE_all'], 0.0, decimal=5)
        assert_almost_equal(infos['MPJPE_single'], 0.0, decimal=5)
        assert_almost_equal(infos['MPJPE_interacting'], 0.0, decimal=5)
        assert_almost_equal(infos['Handedness_acc'], 1.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
