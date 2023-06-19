_base_ = ['_base_/default_runtime.py']

# model settings
model = dict(
    type='YOLODetector',
    use_syncbn=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmyolo/v0/yolox/'
        'yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_'
        '8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth'),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='PoseBatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1)
        ]),
    backbone=dict(
        type='YOLOXCSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        deepen_factor=0.33,
        widen_factor=0.5,
        in_channels=[256, 512, 1024],
        out_channels=256,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOXPoseHead',
        head_module=dict(
            type='YOLOXPoseHeadModule',
            num_classes=1,
            in_channels=256,
            feat_channels=256,
            widen_factor=0.5,
            stacked_convs=2,
            num_keypoints=17,
            featmap_strides=(8, 16, 32),
            use_depthwise=False,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_pose=dict(
            type='OksLoss',
            metainfo='configs/_base_/datasets/coco.py',
            loss_weight=30.0),
        loss_bbox_aux=dict(
            type='mmdet.L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='PoseSimOTAAssigner',
            center_radius=2.5,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            oks_calculator=dict(
                type='OksLoss', metainfo='configs/_base_/datasets/coco.py'))),
    test_cfg=dict(
        yolox_style=True,
        multi_label=False,
        score_thr=0.001,
        max_per_img=300,
        nms=dict(type='nms', iou_threshold=0.65)))

# data related
img_scale = (640, 640)

# pipelines
pre_transform = [
    dict(type='mmpose.LoadImage', backend_args=_base_.backend_args),
    dict(type='PoseToDetConverter')
]

train_pipeline_stage1 = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.75, 1.0),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='FilterDetPoseAnnotations', keep_empty=False),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='FilterDetPoseAnnotations', keep_empty=False),
    dict(type='PackDetPoseInputs')
]

test_pipeline = [
    *pre_transform,
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip_indices'))
]

# dataset settings
dataset_type = 'CocoDataset'
data_mode = 'bottomup'
data_root = 'data/coco/'

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        data_root=data_root,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline_stage1))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        data_root=data_root,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='mmpose.CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    score_mode='bbox')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# optimizer
base_lr = 0.004
max_epochs = 300
num_last_epochs = 20
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last num_last_epochs epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

# runtime
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])

auto_scale_lr = dict(base_batch_size=256)
