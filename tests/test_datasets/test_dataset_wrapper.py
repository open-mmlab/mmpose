# Copyright (c) OpenMMLab. All rights reserved.
from mmcv import Config

from mmpose.datasets.builder import build_dataset


def test_concat_dataset():
    # build COCO-like dataset config
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/coco.py').dataset_info

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

    dataset_cfg = dict(
        type='TopDownCocoDataset',
        ann_file='tests/data/coco/test_coco.json',
        img_prefix='tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=[],
        dataset_info=dataset_info)

    dataset = build_dataset(dataset_cfg)

    # Case 1: build ConcatDataset explicitly
    concat_dataset_cfg = dict(
        type='ConcatDataset', datasets=[dataset_cfg, dataset_cfg])
    concat_dataset = build_dataset(concat_dataset_cfg)
    assert len(concat_dataset) == 2 * len(dataset)

    # Case 2: build ConcatDataset from cfg sequence
    concat_dataset = build_dataset([dataset_cfg, dataset_cfg])
    assert len(concat_dataset) == 2 * len(dataset)

    # Case 3: build ConcatDataset from ann_file sequence
    concat_dataset_cfg = dataset_cfg.copy()
    for key in ['ann_file', 'type', 'img_prefix', 'dataset_info']:
        val = concat_dataset_cfg[key]
        concat_dataset_cfg[key] = [val] * 2
    for key in ['num_joints', 'dataset_channel']:
        val = concat_dataset_cfg['data_cfg'][key]
        concat_dataset_cfg['data_cfg'][key] = [val] * 2
    concat_dataset = build_dataset(concat_dataset_cfg)
    assert len(concat_dataset) == 2 * len(dataset)
