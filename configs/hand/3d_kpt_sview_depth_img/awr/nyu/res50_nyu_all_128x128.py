_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/nyu.py'
]
checkpoint_config = dict(interval=1)
# TODO: metric
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

load_from = '/root/mmpose/data/ckpt/new_res50.pth'
used_keypoints_index = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]

channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=36,
    dataset_channel=used_keypoints_index,
    inference_channel=used_keypoints_index)

# model settings
model = dict(
    type='Depthhand3D',  # pretrained=None
    backbone=dict(
        type='AWRResNet',
        depth=50,
        frozen_stages=-1,
        zero_init_residual=False,
        in_channels=1),
    keypoint_head=dict(
        type='AdaptiveWeightingRegression3DHead',
        offset_head_cfg=dict(
            in_channels=256,
            out_channels_vector=42,
            out_channels_scalar=14,
            heatmap_kernel_size=1.0,
        ),
        deconv_head_cfg=dict(
            in_channels=2048,
            out_channels=256,
            depth_size=64,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, 4),
            extra=dict(final_conv_kernel=0, )),
        loss_offset=dict(type='AWRSmoothL1Loss', use_target_weight=False),
        loss_keypoint=dict(type='AWRSmoothL1Loss', use_target_weight=True),
    ),
    train_cfg=dict(use_img_for_head=True),
    test_cfg=dict(use_img_for_head=True, flip_test=False))

data_cfg = dict(
    image_size=[128, 128],
    heatmap_size=[64, 64, 56],
    cube_size=[300, 300, 300],
    heatmap_size_root=64,
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='TopDownGetBboxCenterScale', padding=1.0),
    dict(type='TopDownAffine'),
    dict(type='DepthToTensor'),
    dict(
        type='MultitaskGatherTarget',
        pipeline_list=[
            [
                dict(
                    type='TopDownGenerateTargetRegression',
                    use_zero_mean=True,
                    joint_indices=used_keypoints_index,
                    is_3d=True,
                    normalize_depth=True,
                ),
                dict(
                    type='HandGenerateJointToOffset',
                    heatmap_kernel_size=1.0,
                )
            ],
            [
                dict(
                    type='TopDownGenerateTargetRegression',
                    use_zero_mean=True,
                    joint_indices=used_keypoints_index,
                    is_3d=True,
                    normalize_depth=True,
                )
            ],
        ],
        pipeline_indices=[0, 1],
    ),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs', 'cube_size', 'center_depth', 'focal',
            'princpt', 'image_size', 'joints_cam', 'dataset_channel',
            'joints_uvd'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='TopDownGetBboxCenterScale', padding=1.0),
    dict(type='TopDownAffine'),
    dict(type='DepthToTensor'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs', 'cube_size', 'center_depth', 'focal',
            'princpt', 'image_size', 'joints_cam', 'dataset_channel',
            'joints_uvd'
        ])
]

test_pipeline = val_pipeline

data_root = 'data/nyu'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    shuffle=False,
    train=dict(
        type='NYUHandDataset',
        ann_file=f'{data_root}/annotations/nyu_test_data.json',
        camera_file=f'{data_root}/annotations/nyu_test_camera.json',
        joint_file=f'{data_root}/annotations/nyu_test_joint_3d.json',
        img_prefix=f'{data_root}/images/test/',
        data_cfg=data_cfg,
        use_refined_center=False,
        align_uvd_xyz_direction=True,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='NYUHandDataset',
        ann_file=f'{data_root}/annotations/nyu_test_data.json',
        camera_file=f'{data_root}/annotations/nyu_test_camera.json',
        joint_file=f'{data_root}/annotations/nyu_test_joint_3d.json',
        img_prefix=f'{data_root}/images/test/',
        data_cfg=data_cfg,
        use_refined_center=False,
        align_uvd_xyz_direction=True,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='NYUHandDataset',
        ann_file=f'{data_root}/annotations/nyu_test_data.json',
        camera_file=f'{data_root}/annotations/nyu_test_camera.json',
        joint_file=f'{data_root}/annotations/nyu_test_joint_3d.json',
        img_prefix=f'{data_root}/images/test/',
        data_cfg=data_cfg,
        use_refined_center=False,
        align_uvd_xyz_direction=True,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
