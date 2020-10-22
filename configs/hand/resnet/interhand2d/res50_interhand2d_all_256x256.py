log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metric=['PCK', 'AUC', 'EPE'], key_indicator='AUC')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 50])
total_epochs = 60
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained='models/pytorch/imagenet/resnet50-19c8e357.pth',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process=True,
        shift_heatmap=True,
        unbiased_decoding=False,
        modulate_kernel=11),
    loss_pose=dict(type='JointsMSELoss', use_target_weight=True))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=20, scale_factor=0.3),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]

test_pipeline = val_pipeline

data_root = 'data/interhand2.6m'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='InterHand2DDataset',
        ann_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_train_data.json',
        camera_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_train_camera.json',
        joint_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_train_joint_3d.json',
        img_prefix=f'{data_root}/images/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='InterHand2DDataset',
        ann_file=f'{data_root}/annotations/machine_annot/'
        'InterHand2.6M_val_data.json',
        camera_file=f'{data_root}/annotations/machine_annot/'
        'InterHand2.6M_val_camera.json',
        joint_file=f'{data_root}/annotations/machine_annot/'
        'InterHand2.6M_val_joint_3d.json',
        img_prefix=f'{data_root}/images/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='InterHand2DDataset',
        ann_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_test_data.json',
        camera_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_test_camera.json',
        joint_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_test_joint_3d.json',
        img_prefix=f'{data_root}/images/test/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
)
