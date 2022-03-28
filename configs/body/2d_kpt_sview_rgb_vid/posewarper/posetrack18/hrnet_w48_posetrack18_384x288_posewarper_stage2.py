_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/posetrack18.py'
]
load_from = 'https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage1-08b632aa_20211130.pth'  # noqa: E501
cudnn_benchmark = True
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP', save_best='Total AP')

optimizer = dict(
    type='Adam',
    lr=0.0001,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 15])
total_epochs = 20
log_config = dict(
    interval=100,
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
    type='PoseWarper',
    pretrained=None,
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        frozen_stages=4,
    ),
    concat_tensors=True,
    neck=dict(
        type='PoseWarperNeck',
        in_channels=48,
        out_channels=channel_cfg['num_output_channels'],
        inner_channels=128,
        deform_groups=channel_cfg['num_output_channels'],
        dilations=(3, 6, 12, 18, 24),
        trans_conv_kernel=1,
        res_blocks_cfg=dict(block='BASIC', num_blocks=20),
        offsets_kernel=3,
        deform_conv_kernel=3,
        freeze_trans_layer=True,
        im2col_step=80),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=channel_cfg['num_output_channels'],
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=0, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    use_nms=True,
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.2,
    bbox_file='data/posetrack18/posetrack18_precomputed_boxes/'
    'val_boxes.json',
    # frame_indices_train=[-1, 0],
    frame_index_rand=True,
    frame_index_range=[-2, 2],
    num_adj_frames=1,
    frame_indices_test=[-2, -1, 0, 1, 2],
    # the first weight is the current frame,
    # then on ascending order of frame indices
    frame_weight_train=(0.0, 1.0),
    frame_weight_test=(0.3, 0.1, 0.25, 0.25, 0.1),
)

# take care of orders of the transforms
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=45,
        scale_factor=0.35),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=3),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'frame_weight'
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
        meta_keys=[
            'image_file',
            'center',
            'scale',
            'rotation',
            'bbox_score',
            'flip_pairs',
            'frame_weight',
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/posetrack18'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='TopDownPoseTrack18VideoDataset',
        ann_file=f'{data_root}/annotations/posetrack18_train.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownPoseTrack18VideoDataset',
        ann_file=f'{data_root}/annotations/posetrack18_val.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownPoseTrack18VideoDataset',
        ann_file=f'{data_root}/annotations/posetrack18_val.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
