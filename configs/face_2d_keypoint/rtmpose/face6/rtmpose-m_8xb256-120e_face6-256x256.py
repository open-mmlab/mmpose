_base_ = ['../../../_base_/default_runtime.py']

# lapa coco wflw 300w cofw halpe

# runtime
max_epochs = 120
stage2_num_epochs = 10
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
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
        eta_min=base_lr * 0.005,
        begin=30,
        end=max_epochs,
        T_max=max_epochs - 30,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

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
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/'
            'rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=106,
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
dataset_type = 'LapaDataset'
data_mode = 'topdown'
data_root = 'data/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=80),
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
                p=1.0),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
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
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.5, 1.5],
        rotate_factor=80),
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
                p=0.5),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]

# train dataset
dataset_lapa = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='LaPa/annotations/lapa_trainval.json',
    data_prefix=dict(img='pose/LaPa/'),
    pipeline=[],
)

kpt_68_to_106 = [
    #
    (0, 0),
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
    (5, 10),
    (6, 12),
    (7, 14),
    (8, 16),
    (9, 18),
    (10, 20),
    (11, 22),
    (12, 24),
    (13, 26),
    (14, 28),
    (15, 30),
    (16, 32),
    #
    (17, 33),
    (18, 34),
    (19, 35),
    (20, 36),
    (21, 37),
    #
    (22, 42),
    (23, 43),
    (24, 44),
    (25, 45),
    (26, 46),
    #
    (27, 51),
    (28, 52),
    (29, 53),
    (30, 54),
    #
    (31, 58),
    (32, 59),
    (33, 60),
    (34, 61),
    (35, 62),
    #
    (36, 66),
    (39, 70),
    #
    ((37, 38), 68),
    ((40, 41), 72),
    #
    (42, 75),
    (45, 79),
    #
    ((43, 44), 77),
    ((46, 47), 81),
    #
    (48, 84),
    (49, 85),
    (50, 86),
    (51, 87),
    (52, 88),
    (53, 89),
    (54, 90),
    (55, 91),
    (56, 92),
    (57, 93),
    (58, 94),
    (59, 95),
    (60, 96),
    (61, 97),
    (62, 98),
    (63, 99),
    (64, 100),
    (65, 101),
    (66, 102),
    (67, 103)
]

mapping_halpe = [
    #
    (26, 0),
    (27, 2),
    (28, 4),
    (29, 6),
    (30, 8),
    (31, 10),
    (32, 12),
    (33, 14),
    (34, 16),
    (35, 18),
    (36, 20),
    (37, 22),
    (38, 24),
    (39, 26),
    (40, 28),
    (41, 30),
    (42, 32),
    #
    (43, 33),
    (44, 34),
    (45, 35),
    (46, 36),
    (47, 37),
    #
    (48, 42),
    (49, 43),
    (50, 44),
    (51, 45),
    (52, 46),
    #
    (53, 51),
    (54, 52),
    (55, 53),
    (56, 54),
    #
    (57, 58),
    (58, 59),
    (59, 60),
    (60, 61),
    (61, 62),
    #
    (62, 66),
    (65, 70),
    #
    ((63, 64), 68),
    ((66, 67), 72),
    #
    (68, 75),
    (71, 79),
    #
    ((69, 70), 77),
    ((72, 73), 81),
    #
    (74, 84),
    (75, 85),
    (76, 86),
    (77, 87),
    (78, 88),
    (79, 89),
    (80, 90),
    (81, 91),
    (82, 92),
    (83, 93),
    (84, 94),
    (85, 95),
    (86, 96),
    (87, 97),
    (88, 98),
    (89, 99),
    (90, 100),
    (91, 101),
    (92, 102),
    (93, 103)
]

mapping_wflw = [
    #
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
    (17, 17),
    (18, 18),
    (19, 19),
    (20, 20),
    (21, 21),
    (22, 22),
    (23, 23),
    (24, 24),
    (25, 25),
    (26, 26),
    (27, 27),
    (28, 28),
    (29, 29),
    (30, 30),
    (31, 31),
    (32, 32),
    #
    (33, 33),
    (34, 34),
    (35, 35),
    (36, 36),
    (37, 37),
    (38, 38),
    (39, 39),
    (40, 40),
    (41, 41),
    #
    (42, 42),
    (43, 43),
    (44, 44),
    (45, 45),
    (46, 46),
    (47, 47),
    (48, 48),
    (49, 49),
    (50, 50),
    #
    (51, 51),
    (52, 52),
    (53, 53),
    (54, 54),
    #
    (55, 58),
    (56, 59),
    (57, 60),
    (58, 61),
    (59, 62),
    #
    (60, 66),
    (61, 67),
    (62, 68),
    (63, 69),
    (64, 70),
    (65, 71),
    (66, 72),
    (67, 73),
    #
    (68, 75),
    (69, 76),
    (70, 77),
    (71, 78),
    (72, 79),
    (73, 80),
    (74, 81),
    (75, 82),
    #
    (76, 84),
    (77, 85),
    (78, 86),
    (79, 87),
    (80, 88),
    (81, 89),
    (82, 90),
    (83, 91),
    (84, 92),
    (85, 93),
    (86, 94),
    (87, 95),
    (88, 96),
    (89, 97),
    (90, 98),
    (91, 99),
    (92, 100),
    (93, 101),
    (94, 102),
    (95, 103),
    #
    (96, 104),
    #
    (97, 105)
]

mapping_cofw = [
    #
    (0, 33),
    (2, 38),
    (4, 35),
    (5, 40),
    #
    (1, 46),
    (3, 50),
    (6, 44),
    (7, 48),
    #
    (8, 60),
    (10, 64),
    (12, 62),
    (13, 66),
    #
    (9, 72),
    (11, 68),
    (14, 70),
    (15, 74),
    #
    (18, 57),
    (19, 63),
    (20, 54),
    (21, 60),
    #
    (22, 84),
    (23, 90),
    (24, 87),
    (25, 98),
    (26, 102),
    (27, 93),
    #
    (28, 16)
]
dataset_coco = dict(
    type='CocoWholeBodyFaceDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='detection/coco/train2017/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=kpt_68_to_106)
    ],
)

dataset_wflw = dict(
    type='WFLWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='wflw/annotations/face_landmarks_wflw_train.json',
    data_prefix=dict(img='pose/WFLW/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=mapping_wflw)
    ],
)

dataset_300w = dict(
    type='Face300WDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='300w/annotations/face_landmarks_300w_train.json',
    data_prefix=dict(img='pose/300w/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=kpt_68_to_106)
    ],
)

dataset_cofw = dict(
    type='COFWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='cofw/annotations/cofw_train.json',
    data_prefix=dict(img='pose/COFW/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=mapping_cofw)
    ],
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train_133kpt.json',
    data_prefix=dict(img='pose/Halpe/hico_20160224_det/images/train2015/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=mapping_halpe)
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/lapa.py'),
        datasets=[
            dataset_lapa, dataset_coco, dataset_wflw, dataset_300w,
            dataset_cofw, dataset_halpe
        ],
        pipeline=train_pipeline,
        test_mode=False,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='LaPa/annotations/lapa_test.json',
        data_prefix=dict(img='pose/LaPa/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# test dataset
val_lapa = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='LaPa/annotations/lapa_test.json',
    data_prefix=dict(img='pose/LaPa/'),
    pipeline=[],
)

val_coco = dict(
    type='CocoWholeBodyFaceDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_val_v1.0.json',
    data_prefix=dict(img='detection/coco/val2017/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=kpt_68_to_106)
    ],
)

val_wflw = dict(
    type='WFLWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='wflw/annotations/face_landmarks_wflw_test.json',
    data_prefix=dict(img='pose/WFLW/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=mapping_wflw)
    ],
)

val_300w = dict(
    type='Face300WDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='300w/annotations/face_landmarks_300w_test.json',
    data_prefix=dict(img='pose/300w/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=kpt_68_to_106)
    ],
)

val_cofw = dict(
    type='COFWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='cofw/annotations/cofw_test.json',
    data_prefix=dict(img='pose/COFW/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=mapping_cofw)
    ],
)

val_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_val_v1.json',
    data_prefix=dict(img='detection/coco/val2017/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=106, mapping=mapping_halpe)
    ],
)

test_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/lapa.py'),
        datasets=[val_lapa, val_coco, val_wflw, val_300w, val_cofw, val_halpe],
        pipeline=val_pipeline,
        test_mode=True,
    ))

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='NME', rule='less', max_keep_ckpts=1, interval=1))

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
val_evaluator = dict(
    type='NME',
    norm_mode='keypoint_distance',
)
test_evaluator = val_evaluator
