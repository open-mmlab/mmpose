_base_ = ['../../../_base_/default_runtime.py']

num_keypoints = 4  # CHECK IT PLZ

# runtime
train_cfg = dict(max_epochs=300, val_interval=10)

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
        end=300,
        milestones=[100, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=64)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='val/PCK@0.05', rule='greater'),
)

# custom_hooks = [
# dict(type='PCKAccuracyTrainHook', interval=10, thr=0.05),
# ]

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)


# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/data/rewe_keypoints/'
labels = ["front_right", "rear_right", "front_left", "rear_left"]
symmetries = [{
    "front_left": "rear_right",
    "front_right": "rear_left",
    "rear_left":  "front_right",
    "rear_right": "front_left",
}
]


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
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=512,
        out_channels=num_keypoints,
        #loss= dict(type='KeypointMSELoss', use_target_weight=True),
        loss=dict(
            type='OutputSymmetryLoss',
            labels=labels,
            symmetries=symmetries,
        ),
        decoder=codec,
        labels=labels,
        symmetries = symmetries,
        ),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))



# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    # TODO: plot
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='RandomBrightnessContrast', brightness_limit=[-0.2, 0.2], contrast_limit=[-0.2, 0.2], p=0.4),

            dict(
                type='OneOf',
                transforms=[
                    dict(type='MotionBlur', blur_limit=3, p=0.3),
                    dict(type='MedianBlur', blur_limit=3, p=0.2),
                    dict(type='Blur', blur_limit=3, p=0.2),
                ], p=0.3),

            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.3),
                    dict(type='MultiplicativeNoise', multiplier=(0.9, 1.1), p=0.3),
                ], p=0.4),

            dict(type='HueSaturationValue', hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        labels=labels,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/train.json',
        data_prefix=dict(img='images'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        labels=labels,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/val.json',
        bbox_file='',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        labels=labels,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/test.json',
        bbox_file='',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# evaluators
val_evaluator = [
    #dict(
    #    type='CocoMetric',
    #    ann_file=data_root + "coco/val.json",
    #),
    dict(
        type='PCKAccuracy',
        prefix="val",
        thr=0.01,
        labels=labels,
        symmetries=symmetries,
    ),
    dict(
        type='PCKAccuracy',
        prefix="val",
        thr=0.05,
        labels=labels,
        symmetries=symmetries,
    ),
    dict(
        type='PCKAccuracy',
        thr=0.1,
        prefix="val",
        labels=labels,
        symmetries=symmetries,
    ),
    ]

test_evaluator = [
    dict(
        type='PCKAccuracy',
        prefix="test",
        thr=0.01,
        labels=labels,
        symmetries=symmetries,
    ),
    dict(
        type='PCKAccuracy',
        prefix="test",
        thr=0.05,
        labels=labels,
        symmetries=symmetries,
    ),
    dict(
        type='PCKAccuracy',
        thr=0.1,
        prefix="test",
        labels=labels,
        symmetries=symmetries,
    ),

]
