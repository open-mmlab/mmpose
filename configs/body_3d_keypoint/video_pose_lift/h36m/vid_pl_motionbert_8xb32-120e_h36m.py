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

auto_scale_lr = dict(base_batch_size=512)

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
    remove_root=False,
    reshape_keypoints=False,
    concat_vis=True)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='DSTFormer',
        in_channels=3,
        feat_size=512,
        depth=5,
        num_heads=8,
        mlp_ratio=2,
        seq_len=243,
        att_fuse=True,
    ),
    head=dict(
        type='MotionRegressionHead',
        in_channels=512,
        out_channels=3,
        embedding_size=512,
        loss=dict(type='MPJPELoss'),
        decoder=codec,
    ),
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
    batch_size=32,
    num_workers=2,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        seq_len=1,
        seq_step=1,
        merge_seq=243,
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
