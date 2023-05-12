_base_ = 'mmdet::rtmdet/rtmdet_l_8xb32-300e_coco.py'

input_shape = 320

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.25,
        use_depthwise=True,
    ),
    neck=dict(
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=True,
    ),
    bbox_head=dict(
        in_channels=64,
        feat_channels=64,
        share_conv=False,
        exp_on_reg=False,
        use_depthwise=True,
        num_classes=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(input_shape, input_shape),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(input_shape * 2, input_shape * 2),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(input_shape, input_shape)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(input_shape, input_shape),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(input_shape, input_shape)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_shape, input_shape), keep_ratio=True),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

data_mode = 'topdown'
data_root = 'data/'

train_dataset = dict(
    _delete_=True,
    type='ConcatDataset',
    datasets=[
        dict(
            type='mmpose.OneHand10KDataset',
            data_root=data_root,
            data_mode=data_mode,
            pipeline=train_pipeline,
            ann_file='onehand10k/annotations/onehand10k_train.json',
            data_prefix=dict(img='pose/OneHand10K/')),
        dict(
            type='mmpose.FreiHandDataset',
            data_root=data_root,
            data_mode=data_mode,
            pipeline=train_pipeline,
            ann_file='freihand/annotations/freihand_train.json',
            data_prefix=dict(img='pose/FreiHand/')),
        dict(
            type='mmpose.Rhd2DDataset',
            data_root=data_root,
            data_mode=data_mode,
            pipeline=train_pipeline,
            ann_file='rhd/annotations/rhd_train.json',
            data_prefix=dict(img='pose/RHD/')),
        dict(
            type='mmpose.HalpeHandDataset',
            data_root=data_root,
            data_mode=data_mode,
            pipeline=train_pipeline,
            ann_file='halpe/annotations/halpe_train_v1.json',
            data_prefix=dict(
                img='pose/Halpe/hico_20160224_det/images/train2015/')  # noqa
        )
    ],
    ignore_keys=[
        'CLASSES', 'dataset_keypoint_weights', 'dataset_name', 'flip_indices',
        'flip_pairs', 'keypoint_colors', 'keypoint_id2name',
        'keypoint_name2id', 'lower_body_ids', 'num_keypoints',
        'num_skeleton_links', 'sigmas', 'skeleton_link_colors',
        'skeleton_links', 'upper_body_ids'
    ],
)

test_dataset = dict(
    _delete_=True,
    type='mmpose.OneHand10KDataset',
    data_root=data_root,
    data_mode=data_mode,
    pipeline=test_pipeline,
    ann_file='onehand10k/annotations/onehand10k_test.json',
    data_prefix=dict(img='pose/OneHand10K/'),
)

train_dataloader = dict(dataset=train_dataset)
val_dataloader = dict(dataset=test_dataset)
test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'onehand10k/annotations/onehand10k_test.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

train_cfg = dict(val_interval=1)
