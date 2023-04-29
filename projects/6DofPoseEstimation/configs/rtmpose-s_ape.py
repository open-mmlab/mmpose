default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmpose'),
    logger=dict(type='LoggerHook', interval=1, _scope_='mmpose'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmpose'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        _scope_='mmpose',
        save_best='PCK',
        rule='greater',
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmpose'),
    visualization=dict(
        type='PoseVisualizationHook', enable=False, _scope_='mmpose'))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=300,
        switch_pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(
                type='RandomBBoxTransform',
                shift_factor=0.0,
                scale_factor=[0.75, 1.25],
                rotate_factor=60),
            dict(type='TopdownAffine', input_size=(256, 256)),
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
                        p=0.5)
                ]),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='SimCCLabel',
                    input_size=(256, 256),
                    sigma=(12, 12),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ])
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmpose')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    _scope_='mmpose')
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True,
    num_digits=6,
    _scope_='mmpose')
log_level = 'INFO'
load_from = None
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=10)
val_cfg = dict()
test_cfg = dict()
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/ape/'
dataset_info = dict(from_file='configs/_base_/datasets/linemod.py')
NUM_KEYPOINTS = 8
max_epochs = 300
val_interval = 10
train_batch_size = 32
val_batch_size = 8
stage2_num_epochs = 0
base_lr = 0.0005
randomness = dict(seed=21)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0, end=20),
    dict(
        type='CosineAnnealingLR',
        eta_min=2.5e-05,
        begin=150,
        end=300,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True)
]
auto_scale_lr = dict(base_batch_size=1024)
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(12, 12),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)
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
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=8,
        input_size=(256, 256),
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True),
        decoder=dict(
            type='SimCCLabel',
            input_size=(256, 256),
            sigma=(12, 12),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    test_cfg=dict(flip_test=True))
train_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=30),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='ChannelShuffle', p=0.5),
            dict(type='CLAHE', p=0.5),
            dict(type='ColorJitter', p=0.5),
            dict(
                type='CoarseDropout',
                max_holes=4,
                max_height=0.3,
                max_width=0.3,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5)
        ]),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(256, 256),
            sigma=(12, 12),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.0,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=(256, 256)),
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
                p=0.5)
        ]),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(256, 256),
            sigma=(12, 12),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=32,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='data/ape/',
        metainfo=dict(from_file='configs/_base_/datasets/linemod.py'),
        data_mode='topdown',
        ann_file='linemod_preprocessed_train.json',
        data_prefix=dict(img='rgb/'),
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(
                type='RandomBBoxTransform',
                scale_factor=[0.8, 1.2],
                rotate_factor=30),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='ChannelShuffle', p=0.5),
                    dict(type='CLAHE', p=0.5),
                    dict(type='ColorJitter', p=0.5),
                    dict(
                        type='CoarseDropout',
                        max_holes=4,
                        max_height=0.3,
                        max_width=0.3,
                        min_holes=1,
                        min_height=0.2,
                        min_width=0.2,
                        p=0.5)
                ]),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='SimCCLabel',
                    input_size=(256, 256),
                    sigma=(12, 12),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ]))
val_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/ape/',
        metainfo=dict(from_file='configs/_base_/datasets/linemod.py'),
        data_mode='topdown',
        ann_file='linemod_preprocessed_test.json',
        data_prefix=dict(img='rgb/'),
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/ape/',
        metainfo=dict(from_file='configs/_base_/datasets/linemod.py'),
        data_mode='topdown',
        ann_file='linemod_preprocessed_test.json',
        data_prefix=dict(img='rgb/'),
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='PackPoseInputs')
        ]))
val_evaluator = [
    dict(
        type='CocoMetric', ann_file='data/ape/linemod_preprocessed_test.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[1, 2])
]
test_evaluator = [
    dict(
        type='CocoMetric', ann_file='data/ape/linemod_preprocessed_test.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[1, 2])
]
launcher = 'none'
work_dir = './work_dirs/rtmpose-s_ape'
