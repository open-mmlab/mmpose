default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmpose'),
    logger=dict(type='LoggerHook', interval=1, _scope_='mmpose'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmpose'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        _scope_='mmpose',
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=1),
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
        switch_epoch=210,
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
                    sigma=(5.66, 5.66),
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
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
val_cfg = dict()
test_cfg = dict()
dataset_info = dict(
    dataset_name='coco',
    classes='sjb_rect',
    paper_info=dict(
        author=
        "Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\\'a}r, Piotr and Zitnick, C Lawrence",
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/'),
    keypoint_info=dict({
        0:
        dict(name='angle_30', id=0, color=[51, 153, 255], type='', swap=''),
        1:
        dict(name='angle_60', id=1, color=[51, 153, 255], type='', swap=''),
        2:
        dict(name='angle_90', id=2, color=[51, 153, 255], type='', swap='')
    }),
    skeleton_info=dict({
        0:
        dict(link=('angle_30', 'angle_60'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('angle_60', 'angle_90'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('angle_90', 'angle_30'), id=2, color=[255, 128, 0])
    }),
    joint_weights=[1.0, 1.0, 1.0],
    sigmas=[0.026, 0.025, 0.025])
max_epochs = 210
stage2_num_epochs = 0
base_lr = 0.004
randomness = dict(seed=21)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    loss_scale='dynamic')
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0, end=20),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=105,
        end=210,
        T_max=105,
        by_epoch=True,
        convert_to_iter_based=True)
]
auto_scale_lr = dict(base_batch_size=1024)
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
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
        out_channels=3,
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
            sigma=(5.66, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    test_cfg=dict(flip_test=True))
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/Triangle_140_Keypoint_Dataset/'
train_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=30),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='ChannelShuffle', p=0.5),
            dict(type='CLAHE', p=0.5),
            dict(type='Downscale', scale_min=0.7, scale_max=0.9, p=0.1),
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
            sigma=(5.66, 5.66),
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
            sigma=(5.66, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=128,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='data/Triangle_140_Keypoint_Dataset/',
        metainfo=dict(
            dataset_name='coco',
            classes='sjb_rect',
            paper_info=dict(
                author=
                "Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\\'a}r, Piotr and Zitnick, C Lawrence",
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(
                    name='angle_30',
                    id=0,
                    color=[51, 153, 255],
                    type='',
                    swap=''),
                1:
                dict(
                    name='angle_60',
                    id=1,
                    color=[51, 153, 255],
                    type='',
                    swap=''),
                2:
                dict(
                    name='angle_90',
                    id=2,
                    color=[51, 153, 255],
                    type='',
                    swap='')
            }),
            skeleton_info=dict({
                0:
                dict(link=('angle_30', 'angle_60'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('angle_60', 'angle_90'), id=1, color=[0, 255, 0]),
                2:
                dict(link=('angle_90', 'angle_30'), id=2, color=[255, 128, 0])
            }),
            joint_weights=[1.0, 1.0, 1.0],
            sigmas=[0.026, 0.025, 0.025]),
        data_mode='topdown',
        ann_file='train_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
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
                    dict(
                        type='Downscale', scale_min=0.7, scale_max=0.9, p=0.1),
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
                    sigma=(5.66, 5.66),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ]))
val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/Triangle_140_Keypoint_Dataset/',
        metainfo=dict(
            dataset_name='coco',
            classes='sjb_rect',
            paper_info=dict(
                author=
                "Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\\'a}r, Piotr and Zitnick, C Lawrence",
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(
                    name='angle_30',
                    id=0,
                    color=[51, 153, 255],
                    type='',
                    swap=''),
                1:
                dict(
                    name='angle_60',
                    id=1,
                    color=[51, 153, 255],
                    type='',
                    swap=''),
                2:
                dict(
                    name='angle_90',
                    id=2,
                    color=[51, 153, 255],
                    type='',
                    swap='')
            }),
            skeleton_info=dict({
                0:
                dict(link=('angle_30', 'angle_60'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('angle_60', 'angle_90'), id=1, color=[0, 255, 0]),
                2:
                dict(link=('angle_90', 'angle_30'), id=2, color=[255, 128, 0])
            }),
            joint_weights=[1.0, 1.0, 1.0],
            sigmas=[0.026, 0.025, 0.025]),
        data_mode='topdown',
        ann_file='val_coco.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/Triangle_140_Keypoint_Dataset/',
        metainfo=dict(
            dataset_name='coco',
            classes='sjb_rect',
            paper_info=dict(
                author=
                "Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\\'a}r, Piotr and Zitnick, C Lawrence",
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(
                    name='angle_30',
                    id=0,
                    color=[51, 153, 255],
                    type='',
                    swap=''),
                1:
                dict(
                    name='angle_60',
                    id=1,
                    color=[51, 153, 255],
                    type='',
                    swap=''),
                2:
                dict(
                    name='angle_90',
                    id=2,
                    color=[51, 153, 255],
                    type='',
                    swap='')
            }),
            skeleton_info=dict({
                0:
                dict(link=('angle_30', 'angle_60'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('angle_60', 'angle_90'), id=1, color=[0, 255, 0]),
                2:
                dict(link=('angle_90', 'angle_30'), id=2, color=[255, 128, 0])
            }),
            joint_weights=[1.0, 1.0, 1.0],
            sigmas=[0.026, 0.025, 0.025]),
        data_mode='topdown',
        ann_file='val_coco.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='PackPoseInputs')
        ]))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/Triangle_140_Keypoint_Dataset/val_coco.json')
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/Triangle_140_Keypoint_Dataset/val_coco.json')
launcher = 'none'
