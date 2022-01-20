_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h36m.py'
]
checkpoint_config = dict(interval=20)
evaluation = dict(
    interval=10, metric=['mpjpe', 'p-mpjpe', 'n-mpjpe'], key_indicator='MPJPE')

# optimizer settings
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='exp',
    by_epoch=True,
    gamma=0.98,
)

total_epochs = 200

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * 17,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(3, 3, 3),
        dropout=0.25,
        use_stride_conv=True),
    keypoint_head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=17,
        loss_keypoint=dict(type='MPJPELoss')),
    traj_backbone=dict(
        type='TCN',
        in_channels=2 * 17,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(3, 3, 3),
        dropout=0.25,
        use_stride_conv=True),
    traj_head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=1,
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
        is_trajectory=True),
    loss_semi=dict(
        type='SemiSupervisionLoss',
        joint_parents=[0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
        warmup_iterations=1311376 // 64 // 8 *
        5),  # dataset_size // samples_per_gpu // gpu_num * warmup_epochs
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = 'data/h36m'
labeled_data_cfg = dict(
    num_joints=17,
    seq_len=27,
    seq_frame_interval=1,
    causal=False,
    temporal_padding=True,
    joint_2d_src='gt',
    subset=0.1,
    subjects=['S1'],
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl',
)
unlabeled_data_cfg = dict(
    num_joints=17,
    seq_len=27,
    seq_frame_interval=1,
    causal=False,
    temporal_padding=True,
    joint_2d_src='gt',
    subjects=['S5', 'S6', 'S7', 'S8'],
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl',
    need_2d_label=True)
val_data_cfg = dict(
    num_joints=17,
    seq_len=27,
    seq_frame_interval=1,
    causal=False,
    temporal_padding=True,
    joint_2d_src='gt',
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl')
test_data_cfg = val_data_cfg

train_labeled_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(
        type='RelativeJointRandomFlip',
        item=['input_2d', 'target'],
        flip_cfg=[
            dict(center_mode='static', center_x=0.),
            dict(center_mode='root', center_index=0)
        ],
        visible_item=['input_2d_visible', 'target_visible'],
        flip_prob=0.5),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target',
              ('root_position', 'traj_target')],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position'])
]

train_unlabeled_pipeline = [
    dict(
        type='ImageCoordinateNormalization',
        item=['input_2d', 'target_2d'],
        norm_camera=True),
    dict(
        type='RelativeJointRandomFlip',
        item=['input_2d', 'target_2d'],
        flip_cfg=[
            dict(center_mode='static', center_x=0.),
            dict(center_mode='static', center_x=0.)
        ],
        visible_item='input_2d_visible',
        flip_prob=0.5,
        flip_camera=True),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(type='CollectCameraIntrinsics'),
    dict(
        type='Collect',
        keys=[('input_2d', 'unlabeled_input'),
              ('target_2d', 'unlabeled_target_2d'), 'intrinsics'],
        meta_name='unlabeled_metas',
        meta_keys=['target_image_path', 'flip_pairs'])
]

val_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position'])
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='Body3DSemiSupervisionDataset',
        labeled_dataset=dict(
            type='Body3DH36MDataset',
            ann_file=f'{data_root}/annotation_body3d/fps50/h36m_train.npz',
            img_prefix=f'{data_root}/images/',
            data_cfg=labeled_data_cfg,
            pipeline=train_labeled_pipeline,
            dataset_info={{_base_.dataset_info}}),
        unlabeled_dataset=dict(
            type='Body3DH36MDataset',
            ann_file=f'{data_root}/annotation_body3d/fps50/h36m_train.npz',
            img_prefix=f'{data_root}/images/',
            data_cfg=unlabeled_data_cfg,
            pipeline=train_unlabeled_pipeline,
            dataset_info={{_base_.dataset_info}})),
    val=dict(
        type='Body3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=val_data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=test_data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
