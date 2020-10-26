import copy
import os
import tempfile
from unittest.mock import MagicMock

import json_tricks as json
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS


def load_json_to_output(json_name, prefix=''):
    data = json.load(open(json_name, 'r'))
    outputs = []

    for image_info, anno in zip(data['images'], data['annotations']):
        keypoints = np.array(
            anno['keypoints'], dtype=np.float32).reshape((1, -1, 3))
        box = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(1, -1)
        img_path = []
        img_path[:0] = os.path.join(prefix, image_info['file_name'])
        output = (keypoints, box, img_path, None)
        outputs.append(output)
    return outputs


def test_top_down_COCO_dataset():
    dataset = 'TopDownCocoDataset'
    # test COCO datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

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
        bbox_thr=1.0,
        use_gt_bbox=True,
        image_thr=0.0,
        bbox_file='tests/data/coco/test_coco_det_AP_H_56.json',
    )
    # Test det bbox
    data_cfg_copy = copy.deepcopy(data_cfg)
    data_cfg_copy['use_gt_bbox'] = False
    _ = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True

    image_id = 785
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 4
    _ = custom_dataset[0]


def test_top_down_PoseTrack18_dataset():
    dataset = 'TopDownPoseTrack18Dataset'
    # test PoseTrack datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

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
        bbox_thr=1.0,
        use_gt_bbox=True,
        image_thr=0.0,
        bbox_file='tests/data/posetrack18/'
        'test_posetrack18_human_detections.json',
    )
    # Test det bbox
    data_cfg_copy = copy.deepcopy(data_cfg)
    data_cfg_copy['use_gt_bbox'] = False
    _ = dataset_class(
        ann_file='tests/data/posetrack18/test_posetrack18.json',
        img_prefix='tests/data/posetrack18/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/posetrack18/test_posetrack18.json',
        img_prefix='tests/data/posetrack18/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/posetrack18/test_posetrack18.json',
        img_prefix='tests/data/posetrack18/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True

    image_id = 10128340000
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]


def test_top_down_CrowdPose_dataset():
    dataset = 'TopDownCrowdPoseDataset'
    # test CrowdPose datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=14,
        dataset_joints=14,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

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
        bbox_thr=1.0,
        use_gt_bbox=True,
        image_thr=0.0,
        bbox_file='tests/data/crowdpose/test_crowdpose_det_AP_40.json',
    )
    # Test det bbox
    data_cfg_copy = copy.deepcopy(data_cfg)
    data_cfg_copy['use_gt_bbox'] = False
    _ = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True

    image_id = 103319
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]


def test_top_down_COCO_wholebody_dataset():
    dataset = 'TopDownCocoWholeBodyDataset'
    # test COCO datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=133,
        dataset_joints=133,
        dataset_channel=[
            list(range(133)),
        ],
        inference_channel=list(range(133)))

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
        bbox_thr=1.0,
        use_gt_bbox=True,
        image_thr=0.0,
        bbox_file='tests/data/coco/test_coco_det_AP_H_56.json',
    )
    # Test det bbox
    data_cfg_copy = copy.deepcopy(data_cfg)
    data_cfg_copy['use_gt_bbox'] = False
    _ = dataset_class(
        ann_file='tests/data/coco/test_coco_wholebody.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/coco/test_coco_wholebody.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/coco/test_coco_wholebody.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True

    image_id = 785
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 4
    _ = custom_dataset[0]


def test_top_down_OCHuman_dataset():
    dataset = 'TopDownOCHumanDataset'
    # test OCHuman datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

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
        bbox_thr=1.0,
        use_gt_bbox=True,
        image_thr=0.0,
        bbox_file='',
    )

    with pytest.raises(AssertionError):
        # Test det bbox
        data_cfg_copy = copy.deepcopy(data_cfg)
        data_cfg_copy['use_gt_bbox'] = False
        _ = dataset_class(
            ann_file='tests/data/ochuman/test_ochuman.json',
            img_prefix='tests/data/ochuman/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=True)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/ochuman/test_ochuman.json',
        img_prefix='tests/data/ochuman/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True

    image_id = 1
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]


def test_top_down_OneHand10K_dataset():
    dataset = 'TopDownOneHand10KDataset'
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

    outputs = load_json_to_output('tests/data/onehand10k/test_onehand10k.json')
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK', 'EPE', 'AUC'])
        assert_almost_equal(infos['PCK'], 1.0)
        assert_almost_equal(infos['AUC'], 0.95)
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_FreiHand_dataset():
    dataset = 'TopDownFreiHandDataset'
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

    outputs = load_json_to_output('tests/data/freihand/test_freihand.json')
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK', 'EPE', 'AUC'])
        assert_almost_equal(infos['PCK'], 1.0)
        assert_almost_equal(infos['AUC'], 0.95)
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_Panoptic_dataset():
    dataset = 'TopDownPanopticDataset'
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

    outputs = load_json_to_output('tests/data/panoptic/test_panoptic.json',
                                  'tests/data/panoptic/')
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir,
                                        ['PCKh', 'EPE', 'AUC'])
        assert_almost_equal(infos['PCKh'], 1.0)
        assert_almost_equal(infos['AUC'], 0.95)
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_MPII_dataset():
    dataset = 'TopDownMpiiDataset'
    # test COCO datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=16,
        dataset_joints=16,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ])

    data_cfg = dict(
        image_size=[256, 256],
        heatmap_size=[64, 64],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
    )

    # Test det bbox
    data_cfg_copy = copy.deepcopy(data_cfg)
    custom_dataset = dataset_class(
        ann_file='tests/data/mpii/test_mpii.json',
        img_prefix='tests/data/mpii/',
        data_cfg=data_cfg_copy,
        pipeline=[])

    assert len(custom_dataset) == 5
    _ = custom_dataset[0]


def test_top_down_MPII_TRB_dataset():
    dataset = 'TopDownMpiiTrbDataset'
    # test MPII TRB datasets
    dataset_class = DATASETS.get(dataset)

    channel_cfg = dict(
        num_output_channels=40,
        dataset_joints=40,
        dataset_channel=[list(range(40))],
        inference_channel=list(range(40)))

    data_cfg = dict(
        image_size=[256, 256],
        heatmap_size=[64, 64],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'])

    data_cfg_copy = copy.deepcopy(data_cfg)
    _ = dataset_class(
        ann_file='tests/data/mpii/test_mpii_trb.json',
        img_prefix='tests/data/mpii/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/mpii/test_mpii_trb.json',
        img_prefix='tests/data/mpii/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
    _ = custom_dataset[0]


def test_top_down_AIC_dataset():
    dataset = 'TopDownAicDataset'
    # test AIC datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=14,
        dataset_joints=14,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

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
        bbox_thr=1.0,
        use_gt_bbox=True,
        image_thr=0.0,
        bbox_file='')

    with pytest.raises(AssertionError):
        # Test det bbox
        data_cfg_copy = copy.deepcopy(data_cfg)
        data_cfg_copy['use_gt_bbox'] = False
        _ = dataset_class(
            ann_file='tests/data/aic/test_aic.json',
            img_prefix='tests/data/aic/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=True)

        _ = dataset_class(
            ann_file='tests/data/aic/test_aic.json',
            img_prefix='tests/data/aic/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/aic/test_aic.json',
        img_prefix='tests/data/aic/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True

    image_id = 1
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]
