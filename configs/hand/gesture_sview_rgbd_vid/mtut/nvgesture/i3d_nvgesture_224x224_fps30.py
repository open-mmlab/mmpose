_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/nvgesture.py'
]

checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metric='AP', save_best='AP_rgb')

optimizer = dict(
    type='SGD',
    lr=1e-2,
    momentum=0.9,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.1, step=[30, 50])
total_epochs = 75
log_config = dict(interval=10)

custom_hooks = [dict(type='ModelSetEpochHook')]

model = dict(
    type='GestureRecognizer',
    modality=['rgb', 'depth'],
    pretrained=dict(
        rgb='https://github.com/hassony2/kinetics_i3d_pytorch/'
        'raw/master/model/model_rgb.pth',
        depth='https://github.com/hassony2/kinetics_i3d_pytorch/'
        'raw/master/model/model_rgb.pth',
    ),
    backbone=dict(
        rgb=dict(
            type='I3D',
            in_channels=3,
            expansion=1,
        ),
        depth=dict(
            type='I3D',
            in_channels=1,
            expansion=1,
        ),
    ),
    cls_head=dict(
        type='MultiModalSSAHead',
        num_classes=25,
    ),
    train_cfg=dict(
        beta=2,
        lambda_=5e-3,
        ssa_start_epoch=61,
    ),
    test_cfg=dict(),
)

data_cfg = dict(
    video_size=[320, 240],
    modality=['rgb', 'depth'],
)

train_pipeline = [
    dict(type='LoadVideoFromFile'),
    dict(type='ModalWiseChannelProcess'),
    dict(type='CropValidClip'),
    dict(type='TemporalPooling', length=64, ref_fps=30),
    dict(type='ResizeGivenShortEdge', length=256),
    dict(type='RandomAlignedSpatialCrop', length=224),
    dict(type='GestureRandomFlip'),
    dict(type='MultiModalVideoToTensor'),
    dict(
        type='VideoNormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect', keys=['video', 'label'], meta_keys=['fps',
                                                            'modality']),
]

val_pipeline = [
    dict(type='LoadVideoFromFile'),
    dict(type='ModalWiseChannelProcess'),
    dict(type='CropValidClip'),
    dict(type='TemporalPooling', length=-1, ref_fps=30),
    dict(type='ResizeGivenShortEdge', length=256),
    dict(type='CenterSpatialCrop', length=224),
    dict(type='MultiModalVideoToTensor'),
    dict(
        type='VideoNormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect', keys=['video', 'label'], meta_keys=['fps',
                                                            'modality']),
]

test_pipeline = val_pipeline

data_root = 'data/nvgesture'
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=6),
    test_dataloader=dict(samples_per_gpu=6),
    train=dict(
        type='NVGestureDataset',
        ann_file=f'{data_root}/annotations/'
        'nvgesture_train_correct_cvpr2016_v2.lst',
        vid_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='NVGestureDataset',
        ann_file=f'{data_root}/annotations/'
        'nvgesture_test_correct_cvpr2016_v2.lst',
        vid_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        test_mode=True,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='NVGestureDataset',
        ann_file=f'{data_root}/annotations/'
        'nvgesture_test_correct_cvpr2016_v2.lst',
        vid_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
        dataset_info={{_base_.dataset_info}}))
