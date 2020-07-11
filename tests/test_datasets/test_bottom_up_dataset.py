import copy

from mmpose.datasets import DATASETS


def test_bottom_up_COCO_dataset():
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
        scale_aware_sigma=False)

    # Test det bbox
    data_cfg_copy = copy.deepcopy(data_cfg)
    custom_dataset = dataset_class(
        ann_file='tests/data/test_coco.json',
        img_prefix='tests/data/',
        data_cfg=data_cfg_copy,
        pipeline=[])

    assert custom_dataset.num_images == 4
