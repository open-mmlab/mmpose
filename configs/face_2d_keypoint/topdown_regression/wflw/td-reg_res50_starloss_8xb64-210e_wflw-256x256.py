_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings for keypoint labels
codec = dict(type='RegressionLabel', input_size=(256, 256))

# codec_star setting for predicted heatmap
codec_star = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(32, 32), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    # replace GlobalAveragePooling with FeatureMapProcessor
    # to obtain heatmap output
    # neck=dict(type='GlobalAveragePooling'),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    # using STARLoss with STARHead
    head=dict(
        type='STARHead',
        in_channels=2048,
        out_channels=98,
        deconv_out_channels=None,
        loss=dict(type='STARLoss', use_target_weight=True),
        decoder=codec_star),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        shift_coords=True,
    ))

# base dataset settings
dataset_type = 'WFLWDataset'
data_mode = 'topdown'
data_root = 'data/wflw/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# dataloaders
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/face_landmarks_wflw_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/face_landmarks_wflw_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(checkpoint=dict(save_best='NME', rule='less'))

# evaluators
val_evaluator = dict(
    type='NME',
    norm_mode='keypoint_distance',
)
test_evaluator = val_evaluator
