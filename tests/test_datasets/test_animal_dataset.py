import copy
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS


def convert_db_to_output(db, batch_size=2, keys=None, is_3d=False):
    outputs = []
    len_db = len(db)
    for i in range(0, len_db, batch_size):
        if is_3d:
            keypoints = np.stack([
                db[j]['joints_3d'].reshape((-1, 3))
                for j in range(i, min(i + batch_size, len_db))
            ])
        else:
            keypoints = np.stack([
                np.hstack([
                    db[j]['joints_3d'].reshape((-1, 3))[:, :2],
                    db[j]['joints_3d_visible'].reshape((-1, 3))[:, :1]
                ]) for j in range(i, min(i + batch_size, len_db))
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

        if keys is not None:
            keys = keys if isinstance(keys, list) else [keys]
            for key in keys:
                output[key] = [
                    db[j][key] for j in range(i, min(i + batch_size, len_db))
                ]

        outputs.append(output)

    return outputs


def test_animal_horse10_dataset():
    dataset = 'AnimalHorse10Dataset'
    dataset_class = DATASETS.get(dataset)

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
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/horse10/test_horse10.json',
        img_prefix='tests/data/horse10/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 3
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK'])
        assert_almost_equal(infos['PCK'], 1.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_animal_fly_dataset():
    dataset = 'AnimalFlyDataset'
    dataset_class = DATASETS.get(dataset)

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
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/fly/test_fly.json',
        img_prefix='tests/data/fly/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK'])
        assert_almost_equal(infos['PCK'], 1.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_animal_locust_dataset():
    dataset = 'AnimalLocustDataset'
    dataset_class = DATASETS.get(dataset)

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
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/locust/test_locust.json',
        img_prefix='tests/data/locust/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK'])
        assert_almost_equal(infos['PCK'], 1.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_animal_zebra_dataset():
    dataset = 'AnimalZebraDataset'
    dataset_class = DATASETS.get(dataset)

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
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/zebra/test_zebra.json',
        img_prefix='tests/data/zebra/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK'])
        assert_almost_equal(infos['PCK'], 1.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_animal_ATRW_dataset():
    dataset = 'AnimalATRWDataset'
    dataset_class = DATASETS.get(dataset)

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
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/atrw/test_atrw.json',
        img_prefix='tests/data/atrw/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK'])


def test_animal_Macaque_dataset():
    dataset = 'AnimalMacaqueDataset'
    dataset_class = DATASETS.get(dataset)

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
        pipeline=[],
        test_mode=True)

    custom_dataset = dataset_class(
        ann_file='tests/data/macaque/test_macaque.json',
        img_prefix='tests/data/macaque/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    assert custom_dataset.test_mode is False
    assert custom_dataset.num_images == 2
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK'])
