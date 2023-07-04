_base_ = ['../../../_base_/default_runtime.py']

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=80, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))

# learning policy
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.975, end=80, by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=1024)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=20),
)

# codec settings
codec = dict(
    type='VideoPoseLifting',
    num_keypoints=17,
    zero_center=True,
    root_index=0,
    remove_root=False)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='TCN',
        in_channels=2 * 17,
        stem_channels=1024,
        num_blocks=4,
        kernel_sizes=(3, 3, 3, 3, 3),
        dropout=0.25,
        use_stride_conv=True,
    ),
    head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=17,
        loss=dict(type='MPJPELoss'),
        decoder=codec,
    ))

# base dataset settings
dataset_type = 'Human36mDataset'
data_root = 'data/h36m/'

# pipelines
train_pipeline = [
    dict(
        type='RandomFlipAroundRoot',
        keypoints_flip_cfg=dict(),
        target_flip_cfg=dict(),
    ),
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root'))
]
val_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root'))
]

# data loaders
train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_train.npz',
        seq_len=243,
        causal=False,
        pad_video_seq=True,
        camera_param_file='annotation_body3d/cameras.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        seq_len=243,
        causal=False,
        pad_video_seq=True,
        camera_param_file='annotation_body3d/cameras.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        test_mode=True,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='MPJPE', mode='mpjpe'),
    dict(type='MPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator
