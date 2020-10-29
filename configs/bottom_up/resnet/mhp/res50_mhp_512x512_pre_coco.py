log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=5)
evaluation = dict(interval=100, metric='mAP', key_indicator='AP')

optimizer = dict(
    type='Adam',
    lr=0.0015,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60, 78])
total_epochs = 90
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    dataset_joints=16,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    ])

data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='BottomUp',
    # pretrained='models/pytorch/imagenet/resnet50-19c8e357.pth',
    pretrained='models/pytorch/coco/res50_coco_512x512-5521bead_20200816.pth',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='BottomUpSimpleHead',
        in_channels=2048,
        num_joints=16,
        tag_per_joint=True,
        with_ae_loss=[True]),
    train_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        img_size=data_cfg['image_size']),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True],
        with_ae=[True],
        project2image=True,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True),
    loss_pose=dict(
        type='MultiLossFactory',
        num_joints=16,
        num_stages=1,
        ae_loss_type='exp',
        with_ae_loss=[True],
        push_loss_factor=[0.001],
        pull_loss_factor=[0.001],
        with_heatmaps_loss=[True],
        heatmaps_loss_factor=[1.0],
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=30,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/coco'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type='BottomUpMhpDataset',
        ann_file='/mnt/lustre/share/zhushimeng/lv-mhp/train_new_wi_area.json',
        img_prefix='/mnt/lustre/share/jinsheng/LV-MHP-v2/train/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='BottomUpMhpDataset',
        ann_file='/mnt/lustre/share/zhushimeng/lv-mhp/val_new_wi_area.json',
        img_prefix='/mnt/lustre/share/jinsheng/LV-MHP-v2/val/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='BottomUpMhpDataset',
        ann_file='/mnt/lustre/share/zhushimeng/lv-mhp/val_new_wi_area.json',
        img_prefix='/mnt/lustre/share/jinsheng/LV-MHP-v2/val/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
)
