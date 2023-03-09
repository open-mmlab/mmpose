# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS


def convert_coco_to_output(coco, is_wholebody=False):
    outputs = []
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

        outputs.append(output)

    return outputs


def test_bottom_up_COCO_dataset_compatibility():
    dataset = 'BottomUpCocoDataset'
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

    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/coco/test_coco.json',
            img_prefix='tests/data/coco/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=False)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/coco/test_coco.json',
            img_prefix='tests/data/coco/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=True)

    assert custom_dataset.num_images == 4
    _ = custom_dataset[0]
    assert custom_dataset.dataset_name == 'coco'

    outputs = convert_coco_to_output(custom_dataset.coco)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_bottom_up_CrowdPose_dataset_compatibility():
    dataset = 'BottomUpCrowdPoseDataset'
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

    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/crowdpose/test_crowdpose.json',
            img_prefix='tests/data/crowdpose/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=False)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/crowdpose/test_crowdpose.json',
            img_prefix='tests/data/crowdpose/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=True)

    image_id = 103319
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]
    assert custom_dataset.dataset_name == 'crowdpose'

    outputs = convert_coco_to_output(custom_dataset.coco)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_bottom_up_MHP_dataset_compatibility():
    dataset = 'BottomUpMhpDataset'
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

    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/mhp/test_mhp.json',
            img_prefix='tests/data/mhp/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=False)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/mhp/test_mhp.json',
            img_prefix='tests/data/mhp/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=True)

    image_id = 2889
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]
    assert custom_dataset.dataset_name == 'mhp'

    outputs = convert_coco_to_output(custom_dataset.coco)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_bottom_up_AIC_dataset_compatibility():
    dataset = 'BottomUpAicDataset'
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

    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/aic/test_aic.json',
            img_prefix='tests/data/aic/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=False)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/aic/test_aic.json',
            img_prefix='tests/data/aic/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=True)

    image_id = 1
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]

    outputs = convert_coco_to_output(custom_dataset.coco)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_bottom_up_COCO_wholebody_dataset_compatibility():
    dataset = 'BottomUpCocoWholeBodyDataset'
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

    with pytest.warns(DeprecationWarning):
        _ = dataset_class(
            ann_file='tests/data/coco/test_coco_wholebody.json',
            img_prefix='tests/data/coco/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=False)

    with pytest.warns(DeprecationWarning):
        custom_dataset = dataset_class(
            ann_file='tests/data/coco/test_coco_wholebody.json',
            img_prefix='tests/data/coco/',
            data_cfg=data_cfg,
            pipeline=[],
            test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'coco_wholebody'

    image_id = 785
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 4
    _ = custom_dataset[0]

    outputs = convert_coco_to_output(custom_dataset.coco, is_wholebody=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')
