_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/panoptic_body3d.py'
]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP', save_best='mAP')

optimizer = dict(
    type='Adam',
    lr=0.0001,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 9])
total_epochs = 15
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

space_size = [8000, 8000, 2000]
space_center = [0, -500, 800]
cube_size = [80, 80, 20]
sub_space_size = [2000, 2000, 2000]
sub_cube_size = [64, 64, 64]
image_size = [960, 512]
heatmap_size = [240, 128]
num_joints = 15

train_data_cfg = dict(
    image_size=image_size,
    heatmap_size=[heatmap_size],
    num_joints=num_joints,
    seq_list=[
        '160422_ultimatum1', '160224_haggling1', '160226_haggling1',
        '161202_haggling1', '160906_ian1', '160906_ian2', '160906_ian3',
        '160906_band1', '160906_band2'
    ],
    cam_list=[(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)],
    num_cameras=5,
    seq_frame_interval=3,
    subset='train',
    root_id=2,
    max_num=10,
    space_size=space_size,
    space_center=space_center,
    cube_size=cube_size,
)

test_data_cfg = train_data_cfg.copy()
test_data_cfg.update(
    dict(
        seq_list=[
            '160906_pizza1',
            '160422_haggling1',
            '160906_ian5',
            '160906_band4',
        ],
        seq_frame_interval=12,
        subset='validation'))

# model settings
backbone = dict(
    type='AssociativeEmbedding',
    pretrained=None,
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='DeconvHead',
        in_channels=2048,
        out_channels=num_joints,
        num_deconv_layers=3,
        num_deconv_filters=(256, 256, 256),
        num_deconv_kernels=(4, 4, 4),
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=15,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[False],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0],
        )),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=num_joints,
        nms_kernel=None,
        nms_padding=None,
        tag_per_joint=None,
        max_num_people=None,
        detection_threshold=None,
        tag_threshold=None,
        use_detection_val=None,
        ignore_too_much=None,
    ))

model = dict(
    type='DetectAndRegress',
    backbone=backbone,
    pretrained='checkpoints/resnet_50_deconv.pth.tar',
    human_detector=dict(
        type='VoxelCenterDetector',
        image_size=image_size,
        heatmap_size=heatmap_size,
        space_size=space_size,
        cube_size=cube_size,
        space_center=space_center,
        center_net=dict(type='V2VNet', input_channels=15, output_channels=1),
        center_head=dict(
            type='CuboidCenterHead',
            space_size=space_size,
            space_center=space_center,
            cube_size=cube_size,
            max_num=10,
            max_pool_kernel=3),
        train_cfg=dict(dist_threshold=500.0),
        test_cfg=dict(center_threshold=0.3),
    ),
    pose_regressor=dict(
        type='VoxelSinglePose',
        image_size=image_size,
        heatmap_size=heatmap_size,
        sub_space_size=sub_space_size,
        sub_cube_size=sub_cube_size,
        num_joints=15,
        pose_net=dict(type='V2VNet', input_channels=15, output_channels=15),
        pose_head=dict(type='CuboidPoseHead', beta=100.0)))

train_pipeline = [
    dict(
        type='MultiItemProcess',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='BottomUpRandomAffine',
                rot_factor=0,
                scale_factor=[1.0, 1.0],
                scale_type='long',
                trans_factor=0),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=[
            'joints_3d', 'joints_3d_visible', 'ann_info', 'roots_3d',
            'num_persons', 'sample_id'
        ]),
    dict(type='GenerateVoxel3DHeatmapTarget', sigma=200.0, joint_indices=[2]),
    dict(
        type='Collect',
        keys=['img', 'targets_3d'],
        meta_keys=[
            'num_persons', 'joints_3d', 'camera', 'center', 'scale',
            'joints_3d_visible', 'roots_3d'
        ]),
]

val_pipeline = [
    dict(
        type='MultiItemProcess',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='BottomUpRandomAffine',
                rot_factor=0,
                scale_factor=[1.0, 1.0],
                scale_type='long',
                trans_factor=0),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=[
            'joints_3d', 'joints_3d_visible', 'ann_info', 'roots_3d',
            'num_persons', 'sample_id'
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['sample_id', 'camera', 'center', 'scale']),
]

test_pipeline = val_pipeline

data_root = 'data/panoptic/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type='Body3DMviewDirectPanopticDataset',
        ann_file=None,
        img_prefix=data_root,
        data_cfg=train_data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Body3DMviewDirectPanopticDataset',
        ann_file=None,
        img_prefix=data_root,
        data_cfg=test_data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DMviewDirectPanopticDataset',
        ann_file=None,
        img_prefix=data_root,
        data_cfg=test_data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
