_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/schedules/schedule_bs512_ep210.py',
]

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
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
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True),
            upsample=dict(mode='bilinear', align_corners=False)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://msra/hrnetv2_w18',
        )),
    head=dict(
        type='HeatmapHead',
        in_channels=[18, 36, 72, 144],
        input_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        out_channels=21,
        deconv_out_channels=None,
        conv_out_channels=(270, ),
        conv_kernel_sizes=(1, ),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'FreiHandDataset'
data_mode = 'topdown'
data_root = 'data/onehand10k/'

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/onehand10k/':
        'openmmlab:s3://openmmlab/datasets/pose/OneHand10K/',
        'data/onehand10k/':
        'openmmlab:s3://openmmlab/datasets/pose/OneHand10K/'
    }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='TopdownGenerateTarget', target_type='heatmap', encoder=codec),
    dict(type='PackPoseInputs')
]
test_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
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
        ann_file='annotations/onehand10k_train.json',
        data_prefix=dict(img=''),
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
        ann_file='annotations/onehand10k_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(type='PCKAccuracy', thr=0.2)
test_evaluator = val_evaluator
