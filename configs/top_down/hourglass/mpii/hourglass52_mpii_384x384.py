log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metric='mAP')

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
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='HourglassNet',
        num_stacks=1,
    ),
    keypoint_head=dict(
        type='TopDownMultiStageHead',
        in_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        num_stages=1,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
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
    image_size=[384, 384],
    heatmap_size=[96, 96],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    use_gt_bbox=True,
    bbox_file=None,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
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

valid_pipeline = [
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

data_root = 'data/mpii'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_valid.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=valid_pipeline),
    test=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=valid_pipeline),
)
