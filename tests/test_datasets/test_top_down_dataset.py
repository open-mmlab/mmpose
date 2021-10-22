# Copyright (c) OpenMMLab. All rights reserved.
import copy
import tempfile
from unittest.mock import MagicMock

import pytest
from mmcv import Config
from numpy.testing import assert_almost_equal
from tests.utils.data_utils import convert_db_to_output

from mmpose.datasets import DATASETS


def test_top_down_COCO_dataset():
    dataset = 'TopDownCocoDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/coco.py').dataset_info
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
        use_gt_bbox=True,
        det_bbox_thr=0.0,
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
        dataset_info=dataset_info,
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'coco'

    image_id = 785
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 4
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_top_down_MHP_dataset():
    dataset = 'TopDownMhpDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/mhp.py').dataset_info
    # test MHP datasets
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
        det_bbox_thr=0.0,
        bbox_file='',
    )

    # Test det bbox
    with pytest.raises(AssertionError):
        data_cfg_copy = copy.deepcopy(data_cfg)
        data_cfg_copy['use_gt_bbox'] = False

        _ = dataset_class(
            ann_file='tests/data/mhp/test_mhp.json',
            img_prefix='tests/data/mhp/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            dataset_info=dataset_info,
            test_mode=True)

    # Test gt bbox
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

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'mhp'

    image_id = 2889
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_top_down_PoseTrack18_dataset():
    dataset = 'TopDownPoseTrack18Dataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/posetrack18.py').dataset_info
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
        use_gt_bbox=True,
        det_bbox_thr=0.0,
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
        dataset_info=dataset_info,
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/posetrack18/test_posetrack18.json',
        img_prefix='tests/data/posetrack18/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/posetrack18/test_posetrack18.json',
        img_prefix='tests/data/posetrack18/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'posetrack18'

    image_id = 10128340000
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]


def test_top_down_CrowdPose_dataset():
    dataset = 'TopDownCrowdPoseDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/crowdpose.py').dataset_info
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
        use_gt_bbox=True,
        det_bbox_thr=0.0,
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
        dataset_info=dataset_info,
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/crowdpose/test_crowdpose.json',
        img_prefix='tests/data/crowdpose/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'crowdpose'

    image_id = 103319
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_top_down_COCO_wholebody_dataset():
    dataset = 'TopDownCocoWholeBodyDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/coco_wholebody.py').dataset_info
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
        use_gt_bbox=True,
        det_bbox_thr=0.0,
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
        dataset_info=dataset_info,
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/coco/test_coco_wholebody.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    # Test gt bbox
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

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_top_down_halpe_dataset():
    dataset = 'TopDownHalpeDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/halpe.py').dataset_info
    # test Halpe datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=136,
        dataset_joints=136,
        dataset_channel=[
            list(range(136)),
        ],
        inference_channel=list(range(136)))

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
        bbox_file='tests/data/coco/test_coco_det_AP_H_56.json',
    )
    # Test det bbox
    data_cfg_copy = copy.deepcopy(data_cfg)
    data_cfg_copy['use_gt_bbox'] = False
    _ = dataset_class(
        ann_file='tests/data/halpe/test_halpe.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/halpe/test_halpe.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/halpe/test_halpe.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'halpe'

    image_id = 785
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 4
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_top_down_OCHuman_dataset():
    dataset = 'TopDownOCHumanDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/ochuman.py').dataset_info
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
        use_gt_bbox=True,
        det_bbox_thr=0.0,
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
            dataset_info=dataset_info,
            test_mode=True)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/ochuman/test_ochuman.json',
        img_prefix='tests/data/ochuman/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'ochuman'

    image_id = 1
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_top_down_MPII_dataset():
    dataset = 'TopDownMpiiDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/mpii.py').dataset_info
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
        pipeline=[],
        dataset_info=dataset_info,
    )

    assert len(custom_dataset) == 5
    assert custom_dataset.dataset_name == 'mpii'
    _ = custom_dataset[0]


def test_top_down_MPII_TRB_dataset():
    dataset = 'TopDownMpiiTrbDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/mpii_trb.py').dataset_info
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
        dataset_info=dataset_info,
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/mpii/test_mpii_trb.json',
        img_prefix='tests/data/mpii/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'mpii_trb'
    _ = custom_dataset[0]


def test_top_down_AIC_dataset():
    dataset = 'TopDownAicDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/aic.py').dataset_info
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
        use_gt_bbox=True,
        det_bbox_thr=0.0,
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
            dataset_info=dataset_info,
            test_mode=True)

        _ = dataset_class(
            ann_file='tests/data/aic/test_aic.json',
            img_prefix='tests/data/aic/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            dataset_info=dataset_info,
            test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/aic/test_aic.json',
        img_prefix='tests/data/aic/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'aic'

    image_id = 1
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
        assert_almost_equal(infos['AP'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'PCK')


def test_top_down_JHMDB_dataset():
    dataset = 'TopDownJhmdbDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/jhmdb.py').dataset_info
    # test JHMDB datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=15,
        dataset_joints=15,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

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
        bbox_file='')

    with pytest.raises(AssertionError):
        # Test det bbox
        data_cfg_copy = copy.deepcopy(data_cfg)
        data_cfg_copy['use_gt_bbox'] = False
        _ = dataset_class(
            ann_file='tests/data/jhmdb/test_jhmdb_sub1.json',
            img_prefix='tests/data/jhmdb/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            dataset_info=dataset_info,
            test_mode=True)

        _ = dataset_class(
            ann_file='tests/data/jhmdb/test_jhmdb_sub1.json',
            img_prefix='tests/data/jhmdb/',
            data_cfg=data_cfg_copy,
            pipeline=[],
            dataset_info=dataset_info,
            test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/jhmdb/test_jhmdb_sub1.json',
        img_prefix='tests/data/jhmdb/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'jhmdb'

    image_id = 2290001
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 3
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK'])
        assert_almost_equal(infos['Mean PCK'], 1.0)

        infos = custom_dataset.evaluate(outputs, tmpdir, ['tPCK'])
        assert_almost_equal(infos['Mean tPCK'], 1.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'mAP')


def test_top_down_h36m_dataset():
    dataset = 'TopDownH36MDataset'
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/h36m.py').dataset_info
    # test AIC datasets
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
        image_size=[256, 256],
        heatmap_size=[64, 64],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'])

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/h36m/h36m_coco.json',
        img_prefix='tests/data/h36m/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info,
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'h36m'

    image_id = 1
    assert image_id in custom_dataset.img_ids
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, 'EPE')
        assert_almost_equal(infos['EPE'], 0.0)

        with pytest.raises(KeyError):
            _ = custom_dataset.evaluate(outputs, tmpdir, 'AUC')
