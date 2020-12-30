import copy
import tempfile
import json_tricks as json
import os
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS

def convert_db_to_output(db, batch_size=2):
    outputs = []
    len_db = len(db)
    for i in range(0, len_db, batch_size):
        keypoints = np.stack([
            db[j]['joints_3d'].reshape((-1, 3))
            for j in range(i, min(i + batch_size, len_db))
        ])
        image_paths = [
            db[j]['image_file'] for j in range(i, min(i + batch_size, len_db))
        ]
        bbox_ids = [j for j in range(i, min(i + batch_size, len_db))]
        box = np.stack(
            np.array([
                db[j]['center'][0], db[j]['center'][1], db[j]['scale'][0],
                db[j]['scale'][1], db[j]['scale'][0] * db[j]['scale'][1] *
                200 * 200, 1.0
            ],
                     dtype=np.float32)
            for j in range(i, min(i + batch_size, len_db)))

        output = {}
        output['preds'] = keypoints
        output['boxes'] = box
        output['image_paths'] = image_paths
        output['output_heatmap'] = None
        output['bbox_ids'] = bbox_ids

        outputs.append(output)

    return outputs


def test_top_down_OneHand10K_dataset():
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
    _ = dataset_class(
        ann_file='tests/data/onehand10k/test_onehand10k.json',
        img_prefix='tests/data/onehand10k/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

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


def test_top_down_FreiHand_dataset():
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
    _ = dataset_class(
        ann_file='tests/data/freihand/test_freihand.json',
        img_prefix='tests/data/freihand/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

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


def test_top_down_Panoptic_dataset():
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
    _ = dataset_class(
        ann_file='tests/data/panoptic/test_panoptic.json',
        img_prefix='tests/data/panoptic/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

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


def test_top_down_InterHand2D_dataset():
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
    _ = dataset_class(
        ann_file='tests/data/interhand2d/test_interhand2d_data.json',
        camera_file='tests/data/interhand2d/test_interhand2d_camera.json',
        joint_file='tests/data/interhand2d/test_interhand2d_joint_3d.json',
        img_prefix='tests/data/interhand2d/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/interhand2d/test_interhand2d_data.json',
        camera_file='tests/data/interhand2d/test_interhand2d_camera.json',
        joint_file='tests/data/interhand2d/test_interhand2d_joint_3d.json',
        img_prefix='tests/data/interhand2d/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 4
    assert len(custom_dataset.db) == 6

    _ = custom_dataset[0]