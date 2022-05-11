# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from mmcv import Config
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS


def convert_coco_to_output(coco, is_wholebody=False):
    results = []
    for img_id in coco.getImgIds():
        preds = []
        scores = []
        image = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(img_id)
        for ann_id in ann_ids:
            obj = coco.anns[ann_id]
            if is_wholebody:
                keypoints = np.array(obj['keypoints'] + obj['foot_kpts'] +
                                     obj['face_kpts'] + obj['lefthand_kpts'] +
                                     obj['righthand_kpts']).reshape(-1, 3)
            else:
                keypoints = np.array(obj['keypoints']).reshape((-1, 3))
            K = keypoints.shape[0]
            if sum(keypoints[:, 2]) == 0:
                continue
            preds.append(
                np.concatenate((keypoints[:, :2], np.ones(
                    [K, 1]), np.ones([K, 1]) * ann_id),
                               axis=1))
            scores.append(1)
        image_paths = []
        image_paths.append(image['file_name'])

        output = {}
        output['preds'] = np.stack(preds)
        output['scores'] = scores
        output['image_paths'] = image_paths
        output['output_heatmap'] = None

        results.append(output)

    return results


def test_bottom_up_COCO_dataset():
    dataset = 'BottomUpCocoDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/coco.py').dataset_info
    # test COCO datasets
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        dataset_joints=17,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
        ])

    data_cfg = dict(
        image_size=512,
        base_size=256,
        base_sigma=2,
        heatmap_size=[128, 256],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        num_scales=2,
        scale_aware_sigma=False,
        use_nms=True)

    _ = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.dataset_name == 'coco'
    assert custom_dataset.num_images == 4
    _ = custom_dataset[0]

    results = convert_coco_to_output(custom_dataset.coco)

    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='PCK')

    # Test when gt annotations are absent
    del custom_dataset.coco.dataset['annotations']
    with pytest.warns(UserWarning):
        _ = custom_dataset.evaluate(results, metric='mAP')


def test_bottom_up_CrowdPose_dataset():
    dataset = 'BottomUpCrowdPoseDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/crowdpose.py').dataset_info
    # test CrowdPose datasets
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=14,
        dataset_joints=14,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    data_cfg = dict(
        image_size=512,
        base_size=256,
        base_sigma=2,
        heatmap_size=[128, 256],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        num_scales=2,
        scale_aware_sigma=False)

    _ = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.dataset_name == 'crowdpose'

    image_id = 103319
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]

    results = convert_coco_to_output(custom_dataset.coco)
    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='PCK')


def test_bottom_up_MHP_dataset():
    dataset = 'BottomUpMhpDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/mhp.py').dataset_info
    # test MHP datasets
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        dataset_joints=16,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ])

    data_cfg = dict(
        image_size=512,
        base_size=256,
        base_sigma=2,
        heatmap_size=[128],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        num_scales=1,
        scale_aware_sigma=False,
    )

    _ = dataset_class(
        ann_file='tests/data/mhp/test_mhp.json',
        img_prefix='tests/data/mhp/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/mhp/test_mhp.json',
        img_prefix='tests/data/mhp/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.dataset_name == 'mhp'

    image_id = 2889
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]

    results = convert_coco_to_output(custom_dataset.coco)
    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='PCK')


def test_bottom_up_AIC_dataset():
    dataset = 'BottomUpAicDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/aic.py').dataset_info
    # test MHP datasets
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=14,
        dataset_joints=14,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    data_cfg = dict(
        image_size=512,
        base_size=256,
        base_sigma=2,
        heatmap_size=[128],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        num_scales=1,
        scale_aware_sigma=False,
    )

    _ = dataset_class(
        ann_file='tests/data/aic/test_aic.json',
        img_prefix='tests/data/aic/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/aic/test_aic.json',
        img_prefix='tests/data/aic/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.dataset_name == 'aic'

    image_id = 1
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]

    results = convert_coco_to_output(custom_dataset.coco)
    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='PCK')


def test_bottom_up_COCO_wholebody_dataset():
    dataset = 'BottomUpCocoWholeBodyDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/coco_wholebody.py').dataset_info
    # test COCO-wholebody datasets
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=133,
        dataset_joints=133,
        dataset_channel=[
            list(range(133)),
        ],
        inference_channel=list(range(133)))

    data_cfg = dict(
        image_size=512,
        base_size=256,
        base_sigma=2,
        heatmap_size=[128, 256],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        num_scales=2,
        scale_aware_sigma=False,
    )

    _ = dataset_class(
        ann_file='tests/data/coco/test_coco_wholebody.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/coco/test_coco_wholebody.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'coco_wholebody'

    image_id = 785
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 4
    _ = custom_dataset[0]

    results = convert_coco_to_output(custom_dataset.coco, is_wholebody=True)
    infos = custom_dataset.evaluate(results, metric='mAP')
    assert_almost_equal(infos['AP'], 1.0)

    with pytest.raises(KeyError):
        _ = custom_dataset.evaluate(results, metric='PCK')
