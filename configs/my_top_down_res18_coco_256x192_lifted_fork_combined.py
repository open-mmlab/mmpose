log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=50)
evaluation = dict(interval=10, metric=['PCK', 'AUC', 'EPE', 'mAP', 'NME'], key_indicator='AP')
n = 7
optimizer = dict(
    type='Adam',
    lr=1e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[300, 350])
total_epochs = 400
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=n,
    dataset_joints=n,
    dataset_channel=[
        list(range(n)),
    ],
    inference_channel=[
        list(range(n))
])

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=512,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownEnlargeBbox', enlarge_factor=0.1),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.1),
    dict(type='TopDownAffine'),
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
            
            dict(type='HueSaturationValue', hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.4),
        ]),
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
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownEnlargeBbox', enlarge_factor=0.1),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/demodesk_lifted_fork_7p'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='LiftedForkDataset7KP',
        ann_file=f'{data_root}/annotations/forklift_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='LiftedForkDataset7KP',
        ann_file=f'{data_root}/annotations/forklift_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='LiftedForkDataset7KP',
        ann_file=f'{data_root}/annotations/forklift_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
)
