log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metric=['mpjpe', 'p-mpjpe'], save_best='MPJPE')

# optimizer settings
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    step=100000,
    gamma=0.96,
)

total_epochs = 200

log_config = dict(
    interval=50,
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

channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))

# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='HourglassNet',
        downsample_times=2,
        num_stacks=5,
        feat_channel=[1 * 16, 2 * 16, 4 * 16, 8 * 16, 64 * 16]),
    keypoint_head=dict(
        type='TopdownHeatmapMultiStageHead',
        in_channels=channel_cfg['num_output_channels'],
        out_channels=channel_cfg['num_output_channels'],
        num_stages=5,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=0, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=False)),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = 'data/h36m'
data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64, [1, 2, 4, 8, 64]],
    heatmap3d_depth_bound=0.5,
    num_joints=17,
    seq_len=1,
    seq_frame_interval=1,
    causal=True,
    joint_2d_src='gt',
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl',
)

# 3D joint normalization parameters
# From file: '{data_root}/annotation_body3d/fps50/joint3d_rel_stats.pkl'
joint_3d_normalize_param = dict(
    mean=[[-2.55652589e-04, -7.11960570e-03, -9.81433052e-04],
          [-5.65463051e-03, 3.19636009e-01, 7.19329269e-02],
          [-1.01705840e-02, 6.91147892e-01, 1.55352986e-01],
          [2.55651315e-04, 7.11954606e-03, 9.81423866e-04],
          [-5.09729780e-03, 3.27040413e-01, 7.22258095e-02],
          [-9.99656606e-03, 7.08277383e-01, 1.58016408e-01],
          [2.90583676e-03, -2.11363307e-01, -4.74210915e-02],
          [5.67537804e-03, -4.35088906e-01, -9.76974016e-02],
          [5.93884964e-03, -4.91891970e-01, -1.10666618e-01],
          [7.37352083e-03, -5.83948619e-01, -1.31171400e-01],
          [5.41920653e-03, -3.83931702e-01, -8.68145417e-02],
          [2.95964662e-03, -1.87567488e-01, -4.34536934e-02],
          [1.26585822e-03, -1.20170579e-01, -2.82526049e-02],
          [4.67186639e-03, -3.83644089e-01, -8.55125784e-02],
          [1.67648571e-03, -1.97007177e-01, -4.31368364e-02],
          [8.70569015e-04, -1.68664569e-01, -3.73902498e-02]],
    std=[[0.11072244, 0.02238818, 0.07246294],
         [0.15856311, 0.18933832, 0.20880479],
         [0.19179935, 0.24320062, 0.24756193],
         [0.11072181, 0.02238805, 0.07246253],
         [0.15880454, 0.19977188, 0.2147063],
         [0.18001944, 0.25052739, 0.24853247],
         [0.05210694, 0.05211406, 0.06908241],
         [0.09515367, 0.10133032, 0.12899733],
         [0.11742458, 0.12648469, 0.16465091],
         [0.12360297, 0.13085539, 0.16433336],
         [0.14602232, 0.09707956, 0.13952731],
         [0.24347532, 0.12982249, 0.20230181],
         [0.2446877, 0.21501816, 0.23938235],
         [0.13876084, 0.1008926, 0.1424411],
         [0.23687529, 0.14491219, 0.20980829],
         [0.24400695, 0.23975028, 0.25520584]])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=True),
    dict(
        type='Generate3DHeatmapTarget_h36m',
        sigma=2.5,
        max_bound=1,
    ),
    dict(
        type='Collect',
        keys=[('img', 'input'), 'target'],
        meta_name='metas',
        meta_keys=[
            'target_image_path',
            'flip_pairs',
            'root_position',
            'root_position_index',
            'center',
            'scale',
            'rotation',
            'image_file',
            'target_image_path',
            'target_image_paths',
        ])
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
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=True),
    dict(
        type='Collect',
        keys=[('img', 'input'), 'target'],
        meta_name='metas',
        meta_keys=[
            'target_image_path',
            'flip_pairs',
            'root_position',
            'root_position_index',
            'center',
            'scale',
            'rotation',
            'image_file',
            'target_image_path',
            'target_image_paths',
        ])
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='Body3DH36MDataset_E2E',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='Body3DH36MDataset_E2E',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='Body3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline),
)
