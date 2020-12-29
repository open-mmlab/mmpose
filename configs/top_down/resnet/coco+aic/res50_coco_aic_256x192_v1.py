log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=5, metric='mAP', key_indicator='AP')

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
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

num_output_channels = 17 + 14

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=2048,
        out_channels=num_output_channels,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11),
    loss_pose=dict(type='JointsMSELoss', use_target_weight=True))

data_cfg_base = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    det_bbox_thr=0.0,
)

data_cfg_coco = dict(
    num_output_channels=num_output_channels,
    num_joints=17,
    dataset_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ],
    use_gt_bbox=True,
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json')

data_cfg_coco.update(data_cfg_base)

data_cfg_aic = dict(
    num_output_channels=num_output_channels,
    num_joints=14,
    dataset_channel=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    inference_channel=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    use_gt_bbox=True,
    bbox_file='')
data_cfg_aic.update(data_cfg_base)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(type='FuseDataset', keys=['target', 'target_weight']),
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
            'flip_pairs', 'ann_info'
        ]),
]

test_pipeline = val_pipeline

data_root_coco = 'data/coco'
data_root_aic = 'data/aic'

train_coco = dict(
    type='TopDownCocoDataset',
    ann_file=f'{data_root_coco}/annotations/person_keypoints_train2017.json',
    img_prefix=f'{data_root_coco}/train2017/',
    data_cfg=data_cfg_coco,
    pipeline=train_pipeline),

train_aic = dict(
    type='TopDownAicDataset',
    ann_file=f'{data_root_aic}/annotations/aic_train.json',
    img_prefix=f'{data_root_aic}/ai_challenger_keypoint_train_20170902/'
    'keypoint_train_images_20170902/',
    data_cfg=data_cfg_aic,
    pipeline=train_pipeline),

val_coco = dict(
    type='TopDownCocoDataset',
    ann_file=f'{data_root_coco}/annotations/person_keypoints_val2017.json',
    img_prefix=f'{data_root_coco}/val2017/',
    data_cfg=data_cfg_coco,
    pipeline=val_pipeline),

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=[train_coco, train_aic],
    val=val_coco,
    test=val_coco,
)
