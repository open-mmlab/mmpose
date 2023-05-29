_base_ = ['../../../_base_/default_runtime.py']

# coco-hand onehand10k freihand2d rhd2d halpehand

# runtime
max_epochs = 210
stage2_num_epochs = 10
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=21,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'CocoWholeBodyHandDataset'
data_mode = 'topdown'
data_root = 'data/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5],
        rotate_factor=180),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=180),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.2),
            dict(type='MedianBlur', p=0.2),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# train datasets
dataset_coco = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='detection/coco/train2017/'),
    pipeline=[],
)

dataset_onehand10k = dict(
    type='OneHand10KDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='onehand10k/annotations/onehand10k_train.json',
    data_prefix=dict(img='pose/OneHand10K/'),
    pipeline=[],
)

dataset_freihand = dict(
    type='FreiHandDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='freihand/annotations/freihand_train.json',
    data_prefix=dict(img='pose/FreiHand/'),
    pipeline=[],
)

dataset_rhd = dict(
    type='Rhd2DDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='rhd/annotations/rhd_train.json',
    data_prefix=dict(img='pose/RHD/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=21,
            mapping=[
                (0, 0),
                (1, 4),
                (2, 3),
                (3, 2),
                (4, 1),
                (5, 8),
                (6, 7),
                (7, 6),
                (8, 5),
                (9, 12),
                (10, 11),
                (11, 10),
                (12, 9),
                (13, 16),
                (14, 15),
                (15, 14),
                (16, 13),
                (17, 20),
                (18, 19),
                (19, 18),
                (20, 17),
            ])
    ],
)

dataset_halpehand = dict(
    type='HalpeHandDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train_v1.json',
    data_prefix=dict(img='pose/Halpe/hico_20160224_det/images/train2015/'),
    pipeline=[],
)

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(
            from_file='configs/_base_/datasets/coco_wholebody_hand.py'),
        datasets=[
            dataset_coco, dataset_onehand10k, dataset_freihand, dataset_rhd,
            dataset_halpehand
        ],
        pipeline=train_pipeline,
        test_mode=False,
    ))

# test datasets
val_coco = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_val_v1.0.json',
    data_prefix=dict(img='detection/coco/val2017/'),
    pipeline=[],
)

val_onehand10k = dict(
    type='OneHand10KDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='onehand10k/annotations/onehand10k_test.json',
    data_prefix=dict(img='pose/OneHand10K/'),
    pipeline=[],
)

val_freihand = dict(
    type='FreiHandDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='freihand/annotations/freihand_test.json',
    data_prefix=dict(img='pose/FreiHand/'),
    pipeline=[],
)

val_rhd = dict(
    type='Rhd2DDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='rhd/annotations/rhd_test.json',
    data_prefix=dict(img='pose/RHD/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=21,
            mapping=[
                (0, 0),
                (1, 4),
                (2, 3),
                (3, 2),
                (4, 1),
                (5, 8),
                (6, 7),
                (7, 6),
                (8, 5),
                (9, 12),
                (10, 11),
                (11, 10),
                (12, 9),
                (13, 16),
                (14, 15),
                (15, 14),
                (16, 13),
                (17, 20),
                (18, 19),
                (19, 18),
                (20, 17),
            ])
    ],
)

val_halpehand = dict(
    type='HalpeHandDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_val_v1.json',
    data_prefix=dict(img='detection/coco/val2017/'),
    pipeline=[],
)

test_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(
            from_file='configs/_base_/datasets/coco_wholebody_hand.py'),
        datasets=[
            val_coco, val_onehand10k, val_freihand, val_rhd, val_halpehand
        ],
        pipeline=val_pipeline,
        test_mode=True,
    ))

val_dataloader = test_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='AUC', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE')
]
test_evaluator = val_evaluator
