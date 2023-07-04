_base_ = ['../../../_base_/default_runtime.py']

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = None

# optimizer

# learning policy

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
        num_blocks=2,
        kernel_sizes=(3, 3, 3),
        dropout=0.25,
        use_stride_conv=True,
    ),
    head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=17,
        loss=dict(type='MPJPELoss'),
        decoder=codec,
    ),
    traj_backbone=dict(
        type='TCN',
        in_channels=2 * 17,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(3, 3, 3),
        dropout=0.25,
        use_stride_conv=True,
    ),
    traj_head=dict(
        type='TrajectoryRegressionHead',
        in_channels=1024,
        num_joints=1,
        loss=dict(type='MPJPELoss', use_target_weight=True),
        decoder=codec,
    ),
    semi_loss=dict(
        type='SemiSupervisionLoss',
        joint_parents=[0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
        warmup_iterations=1311376 // 64 // 8 * 5),
)

# base dataset settings
dataset_type = 'Human36mDataset'
data_root = 'data/h36m/'

# pipelines
val_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root'))
]

# data loaders
val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        seq_len=27,
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
    dict(type='MPJPE', mode='p-mpjpe'),
    dict(type='MPJPE', mode='n-mpjpe')
]
test_evaluator = val_evaluator
