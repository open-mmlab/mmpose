import copy
from unittest.mock import MagicMock

from mmpose.datasets import DATASETS


def test_COCO_dataset():
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
        bbox_file='tests/data/test_coco_det_AP_H_56.json',
    )
    # Test det bbox
    data_cfg_copy = copy.deepcopy(data_cfg)
    data_cfg_copy['use_gt_bbox'] = False
    _ = dataset_class(
        ann_file='tests/data/test_coco.json',
        img_prefix='tests/data/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=True)

    _ = dataset_class(
        ann_file='tests/data/test_coco.json',
        img_prefix='tests/data/',
        data_cfg=data_cfg_copy,
        pipeline=[],
        test_mode=False)

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/test_coco.json',
        img_prefix='tests/data/',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True

    image_id = 785
    assert image_id in custom_dataset.image_set_index
    assert len(custom_dataset.image_set_index) == 4
