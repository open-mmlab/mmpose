_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_bs512_ep210.py',
]

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=501, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='coco/AP',
        rule='greater'), )

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]

find_unused_parameters = True

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=None,
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose'
            '/pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CocoDatasetV0'
data_mode = 'topdown'
data_root = 'data/coco/'

# file_client_args = dict(backend='disk')
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/coco/': 's3://openmmlab/datasets/detection/coco/',
        'data/coco/': 's3://openmmlab/datasets/detection/coco/'
    }))

# pipelines
train_pipeline = [
    dict(type='LoadImageFromFileV0', file_client_args=file_client_args),
    dict(type='TopDownGetBboxCenterScaleV0', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenterV0', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlipV0', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransformV0',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotationV0',
        rot_factor=40,
        scale_factor=0.5),
    dict(type='TopDownAffineV0'),
    dict(type='ToTensorV0'),
    dict(
        type='NormalizeTensorV0',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetV0', sigma=2),
    dict(type='PackPoseInputsV0')
]
test_pipeline = [
    dict(type='LoadImageFromFileV0', file_client_args=file_client_args),
    dict(type='TopDownGetBboxCenterScaleV0', padding=1.25),
    dict(type='TopDownAffineV0'),
    dict(type='ToTensorV0'),
    dict(
        type='NormalizeTensorV0',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='PackPoseInputsV0')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
