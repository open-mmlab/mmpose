_base_ = ['../../_base_/default_runtime.py']

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=200, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))

# learning policy
param_scheduler = [
    dict(type='StepLR', step_size=100000, gamma=0.96, end=80, by_epoch=False)
]

auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1))

# codec settings
codec = dict(
    type='ImagePoseLifting', num_keypoints=137, root_index=0, remove_root=True)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='TCN',
        in_channels=2 * 137,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(1, 1, 1),
        dropout=0.5,
    ),
    head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=136,
        loss=dict(type='MSELoss'),
        decoder=codec,
    ))

# base dataset settings
dataset_type = 'UBody3dDataset'
data_mode = 'topdown'
data_root = 'data/UBody/'

scenes = [
    'Magic_show', 'Entertainment', 'ConductMusic', 'Online_class', 'TalkShow',
    'Speech', 'Fitness', 'Interview', 'Olympic', 'TVShow', 'Singing',
    'SignLanguage', 'Movie', 'LiveVlog', 'VideoConference'
]

train_datasets = []
val_datasets = []

for scene in scenes:
    train_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=f'annotations/{scene}/train_3dkeypoint_annotation.json',
        seq_len=1,
        causal=True,
        keypoint_2d_src='gt',
        data_prefix=dict(img='images/'),
        pipeline=[])
    val_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=f'annotations/{scene}/val_3dkeypoint_annotation.json',
        data_prefix=dict(img='images/'),
        pipeline=[])
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)

# pipelines
train_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root', 'target_root_index'))
]
val_pipeline = train_pipeline

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/ubody3d.py'),
        datasets=train_datasets,
        pipeline=train_pipeline,
        test_mode=False,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/ubody3d.py'),
        datasets=val_datasets,
        pipeline=val_pipeline,
        test_mode=True,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='SimpleMPJPE', mode='mpjpe'),
    dict(type='SimpleMPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator
