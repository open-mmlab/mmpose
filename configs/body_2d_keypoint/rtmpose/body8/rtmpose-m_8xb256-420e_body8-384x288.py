_base_ = ['../../../_base_/default_runtime.py']

# runtime
max_epochs = 420
stage2_num_epochs = 20
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
        # use cosine lr from 210 to 420 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(288, 384),
    sigma=(6., 6.93),
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
            'rtmposev1/cspnext-m_udp-body7_210e-384x288-b9bc2b57_20230504.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
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
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'CocoDataset'
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
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='PhotometricDistortion'),
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
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
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
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# mapping
aic_coco = [
    (0, 6),
    (1, 8),
    (2, 10),
    (3, 5),
    (4, 7),
    (5, 9),
    (6, 12),
    (7, 14),
    (8, 16),
    (9, 11),
    (10, 13),
    (11, 15),
]

crowdpose_coco = [
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
    (5, 10),
    (6, 11),
    (7, 12),
    (8, 13),
    (9, 14),
    (10, 15),
    (11, 16),
]

mpii_coco = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
]

jhmdb_coco = [
    (3, 6),
    (4, 5),
    (5, 12),
    (6, 11),
    (7, 8),
    (8, 7),
    (9, 14),
    (10, 13),
    (11, 10),
    (12, 9),
    (13, 16),
    (14, 15),
]

halpe_coco = [
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
]

ochuman_coco = [
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
]

posetrack_coco = [
    (0, 0),
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
]

# train datasets
dataset_coco = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='detection/coco/train2017/'),
    pipeline=[],
)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(img='pose/ai_challenge/ai_challenger_keypoint'
                     '_train_20170902/keypoint_train_images_20170902/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=aic_coco)
    ],
)

dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=17, mapping=crowdpose_coco)
    ],
)

dataset_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='pose/MPI/images/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=mpii_coco)
    ],
)

dataset_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/Sub1_train.json',
    data_prefix=dict(img='pose/JHMDB/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=jhmdb_coco)
    ],
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train_v1.json',
    data_prefix=dict(img='pose/Halpe/hico_20160224_det/images/train2015'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=halpe_coco)
    ],
)

dataset_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack18/annotations/posetrack18_train.json',
    data_prefix=dict(img='pose/PoseChallenge2018/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=17, mapping=posetrack_coco)
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
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            dataset_coco,
            dataset_aic,
            dataset_crowdpose,
            dataset_mpii,
            dataset_jhmdb,
            dataset_halpe,
            dataset_posetrack,
        ],
        pipeline=train_pipeline,
        test_mode=False,
    ))

# val datasets
val_coco = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_val2017.json',
    data_prefix=dict(img='detection/coco/val2017/'),
    pipeline=[],
)

val_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_val.json',
    data_prefix=dict(
        img='pose/ai_challenge/ai_challenger_keypoint'
        '_validation_20170911/keypoint_validation_images_20170911/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=aic_coco)
    ],
)

val_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_test.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=17, mapping=crowdpose_coco)
    ],
)

val_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_val.json',
    data_prefix=dict(img='pose/MPI/images/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=mpii_coco)
    ],
)

val_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/Sub1_test.json',
    data_prefix=dict(img='pose/JHMDB/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=jhmdb_coco)
    ],
)

val_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_val_v1.json',
    data_prefix=dict(img='detection/coco/val2017/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=halpe_coco)
    ],
)

val_ochuman = dict(
    type='OCHumanDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ochuman/annotations/'
    'ochuman_coco_format_val_range_0.00_1.00.json',
    data_prefix=dict(img='pose/OCHuman/images/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=17, mapping=ochuman_coco)
    ],
)

val_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack18/annotations/posetrack18_val.json',
    data_prefix=dict(img='pose/PoseChallenge2018/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=17, mapping=posetrack_coco)
    ],
)

val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        bbox_file=f'{data_root}coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='detection/coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            val_coco,
            val_aic,
            val_crowdpose,
            val_mpii,
            val_jhmdb,
            val_halpe,
            val_ochuman,
            val_posetrack,
        ],
        pipeline=val_pipeline,
        test_mode=True,
    ))

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))
# default_hooks = dict(
#     checkpoint=dict(save_best='AUC', rule='greater', max_keep_ckpts=1))

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
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json')
test_evaluator = [
    dict(type='PCKAccuracy', thr=0.1),
    dict(type='AUC'),
    dict(type='EPE')
]
