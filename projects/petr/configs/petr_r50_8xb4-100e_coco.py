_base_ = ['./_base_/default_runtime.py']

# learning policy
max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[80],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
auto_scale_lr = dict(base_batch_size=32)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# model
num_keypoints = 17
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/deformable_' \
    'detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_' \
    'detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
model = dict(
    type='PETR',
    num_queries=300,
    num_feature_levels=4,
    num_keypoints=num_keypoints,
    with_box_refine=True,
    as_two_stage=True,
    init_cfg=dict(
        type='Pretrained',
        checkpoint=checkpoint,
    ),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(  # PetrTransformerDecoder
        num_layers=3,
        num_keypoints=num_keypoints,
        return_intermediate=True,
        layer_cfg=dict(  # PetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformablePoseAttention
                embed_dims=256,
                num_points=num_keypoints,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    hm_encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=1,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                num_levels=1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    kpt_decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=2,
        return_intermediate=True,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                im2col_step=128),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='PETRHead',
        num_classes=1,
        num_keypoints=num_keypoints,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_reg=dict(type='L1Loss', loss_weight=80.0),
        loss_reg_aux=dict(type='L1Loss', loss_weight=70.0),
        loss_oks=dict(
            type='OksLoss',
            metainfo='configs/_base_/datasets/coco.py',
            loss_weight=3.0),
        loss_oks_aux=dict(
            type='OksLoss',
            metainfo='configs/_base_/datasets/coco.py',
            loss_weight=2.0),
        loss_hm=dict(type='mmpose.FocalHeatmapLoss', loss_weight=4.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='KptL1Cost', weight=70.0),
                dict(
                    type='OksCost',
                    metainfo='configs/_base_/datasets/coco.py',
                    weight=7.0)
            ])),
    test_cfg=dict(
        max_per_img=100,
        score_thr=0.0,
    ))

train_pipeline = [
    dict(type='mmpose.LoadImage', backend_args=_base_.backend_args),
    dict(type='PoseToDetConverter'),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='RandomAffine',
        max_rotate_degree=30.0,
        # max_translate_ratio=0.,
        # scaling_ratio_range=(1., 1.),
        # max_shear_degree=0.,
        scaling_ratio_range=(0.75, 1.0),
        border_val=[103.53, 116.28, 123.675],
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=list(zip(range(400, 1401, 8), (1400, ) * 126)),
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=list(zip(range(400, 1401, 8), (1400, ) * 126)),
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterDetPoseAnnotations', keep_empty=False),
    dict(type='GenerateHeatmap'),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

test_pipeline = [
    dict(type='mmpose.LoadImage', backend_args=_base_.backend_args),
    dict(type='PoseToDetConverter'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip_indices'))
]

dataset_type = 'CocoDataset'
data_mode = 'bottomup'
data_root = 'data/coco/'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        data_root=data_root,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
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

val_evaluator = dict(
    type='mmpose.CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='bbox')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))
