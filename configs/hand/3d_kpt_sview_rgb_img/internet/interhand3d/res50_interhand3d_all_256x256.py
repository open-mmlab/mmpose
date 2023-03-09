_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/interhand3d.py'
]
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,
    metric=['MRRPE', 'MPJPE', 'Handedness_acc'],
    save_best='MPJPE_all')

optimizer = dict(
    type='Adam',
    lr=2e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15, 17])
total_epochs = 20
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=42,
    dataset_joints=42,
    dataset_channel=[list(range(42))],
    inference_channel=list(range(42)))

# model settings
model = dict(
    type='Interhand3D',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='Interhand3DHead',
        keypoint_head_cfg=dict(
            in_channels=2048,
            out_channels=21 * 64,
            depth_size=64,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, 4),
        ),
        root_head_cfg=dict(
            in_channels=2048,
            heatmap_size=64,
            hidden_dims=(512, ),
        ),
        hand_type_head_cfg=dict(
            in_channels=2048,
            num_labels=2,
            hidden_dims=(512, ),
        ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        loss_root_depth=dict(type='L1Loss', use_target_weight=True),
        loss_hand_type=dict(type='BCELoss', use_target_weight=True),
    ),
    train_cfg={},
    test_cfg=dict(flip_test=False))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64, 64],
    heatmap3d_depth_bound=400.0,
    heatmap_size_root=64,
    root_depth_bound=400.0,
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='HandRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation',
        rot_factor=45,
        scale_factor=0.25,
        rot_prob=0.6),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='MultitaskGatherTarget',
        pipeline_list=[
            [dict(
                type='Generate3DHeatmapTarget',
                sigma=2.5,
                max_bound=255,
            )], [dict(type='HandGenerateRelDepthTarget')],
            [
                dict(
                    type='RenameKeys',
                    key_pairs=[('hand_type', 'target'),
                               ('hand_type_valid', 'target_weight')])
            ]
        ],
        pipeline_indices=[0, 1, 2],
    ),
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
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
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
            'image_file', 'center', 'scale', 'rotation', 'flip_pairs',
            'heatmap3d_depth_bound', 'root_depth_bound'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/interhand2.6m'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type='InterHand3DDataset',
        ann_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_train_data.json',
        camera_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_train_camera.json',
        joint_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_train_joint_3d.json',
        img_prefix=f'{data_root}/images/train/',
        data_cfg=data_cfg,
        use_gt_root_depth=True,
        rootnet_result_file=None,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='InterHand3DDataset',
        ann_file=f'{data_root}/annotations/machine_annot/'
        'InterHand2.6M_val_data.json',
        camera_file=f'{data_root}/annotations/machine_annot/'
        'InterHand2.6M_val_camera.json',
        joint_file=f'{data_root}/annotations/machine_annot/'
        'InterHand2.6M_val_joint_3d.json',
        img_prefix=f'{data_root}/images/val/',
        data_cfg=data_cfg,
        use_gt_root_depth=True,
        rootnet_result_file=None,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='InterHand3DDataset',
        ann_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_test_data.json',
        camera_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_test_camera.json',
        joint_file=f'{data_root}/annotations/all/'
        'InterHand2.6M_test_joint_3d.json',
        img_prefix=f'{data_root}/images/test/',
        data_cfg=data_cfg,
        use_gt_root_depth=True,
        rootnet_result_file=None,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
