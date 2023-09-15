_base_ = ['../../../_base_/default_runtime.py']

# visualization
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=20, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0002))

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        milestones=[15, 17],
        gamma=0.1,
        by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=128)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='MPJPE_all',
        rule='less',
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=20),
)

# codec settings
codec = dict(
    type='Hand3DHeatmap',
    image_size=[256, 256],
    root_heatmap_size=64,
    heatmap_size=[64, 64, 64],
    sigma=2.5,
    max_bound=255,
    depth_size=64)

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
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(
        type='InternetHead',
        keypoint_head_cfg=dict(
            in_channels=2048,
            out_channels=21 * 64,
            depth_size=codec['depth_size'],
            deconv_out_channels=(256, 256, 256),
            deconv_kernel_sizes=(4, 4, 4),
        ),
        root_head_cfg=dict(
            in_channels=2048,
            heatmap_size=codec['root_heatmap_size'],
            hidden_dims=(512, ),
        ),
        hand_type_head_cfg=dict(
            in_channels=2048,
            num_labels=2,
            hidden_dims=(512, ),
        ),
        decoder=codec),
    test_cfg=dict(flip_test=False))

# base dataset settings
dataset_type = 'InterHand3DDataset'
data_mode = 'topdown'
data_root = 'data/interhand2.6m/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='HandRandomFlip', prob=0.5),
    dict(type='RandomBBoxTransform', rotate_factor=90.0),
    dict(type='TopdownAffine', input_size=codec['image_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'rotation', 'img_shape',
                   'focal', 'principal_pt', 'input_size', 'input_center',
                   'input_scale', 'hand_type', 'hand_type_valid', 'flip',
                   'flip_indices', 'abs_depth'))
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['image_size']),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'rotation', 'img_shape',
                   'focal', 'principal_pt', 'input_size', 'input_center',
                   'input_scale', 'hand_type', 'hand_type_valid', 'flip',
                   'flip_indices', 'abs_depth'))
]

# data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/all/InterHand2.6M_train_data.json',
        camera_param_file='annotations/all/InterHand2.6M_train_camera.json',
        joint_file='annotations/all/InterHand2.6M_train_joint_3d.json',
        use_gt_root_depth=True,
        rootnet_result_file=None,
        data_mode=data_mode,
        data_root=data_root,
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/machine_annot/InterHand2.6M_val_data.json',
        camera_param_file='annotations/machine_annot/'
        'InterHand2.6M_val_camera.json',
        joint_file='annotations/machine_annot/InterHand2.6M_val_joint_3d.json',
        use_gt_root_depth=True,
        rootnet_result_file=None,
        data_mode=data_mode,
        data_root=data_root,
        data_prefix=dict(img='images/val/'),
        pipeline=val_pipeline,
        test_mode=True,
    ))
test_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/all/'
        'InterHand2.6M_test_data.json',
        camera_param_file='annotations/all/'
        'InterHand2.6M_test_camera.json',
        joint_file='annotations/all/'
        'InterHand2.6M_test_joint_3d.json',
        use_gt_root_depth=True,
        rootnet_result_file=None,
        data_mode=data_mode,
        data_root=data_root,
        data_prefix=dict(img='images/test/'),
        pipeline=val_pipeline,
        test_mode=True,
    ))

# evaluators
val_evaluator = [
    dict(type='InterHandMetric', modes=['MPJPE', 'MRRPE', 'HandednessAcc'])
]
test_evaluator = val_evaluator
