_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/mpi_inf_3dhp.py'
]
evaluation = dict(
    interval=10,
    metric=['mpjpe', 'p-mpjpe', '3dpck', '3dauc'],
    save_best='MPJPE')

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
        kernel_sizes=(1, 1, 1),
        dropout=0.5),
    keypoint_head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=16,  # do not predict root joint
        loss_keypoint=dict(type='MSELoss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = 'data/mpi_inf_3dhp'
train_data_cfg = dict(
    num_joints=17,
    seq_len=1,
    seq_frame_interval=1,
    causal=True,
    joint_2d_src='gt',
    need_camera_param=False,
    camera_param_file=f'{data_root}/annotations/cameras_train.pkl',
)
test_data_cfg = dict(
    num_joints=17,
    seq_len=1,
    seq_frame_interval=1,
    causal=True,
    joint_2d_src='gt',
    need_camera_param=False,
    camera_param_file=f'{data_root}/annotations/cameras_test.pkl',
)

# 3D joint normalization parameters
# From file: '{data_root}/annotations/joint3d_rel_stats.pkl'
joint_3d_normalize_param = dict(
    mean=[[1.29798757e-02, -6.14242101e-01, -8.27376088e-02],
          [8.76858608e-03, -3.99992424e-01, -5.62749816e-02],
          [1.96335208e-02, -3.64617227e-01, -4.88267063e-02],
          [2.75206678e-02, -1.95085890e-01, -2.01508894e-02],
          [2.22896982e-02, -1.37878727e-01, -5.51315396e-03],
          [-4.16641282e-03, -3.65152343e-01, -5.43331534e-02],
          [-1.83806493e-02, -1.88053038e-01, -2.78737492e-02],
          [-1.81491930e-02, -1.22997985e-01, -1.15657333e-02],
          [1.02960759e-02, -3.93481284e-03, 2.56594686e-03],
          [-9.82312721e-04, 3.03909927e-01, 6.40930378e-02],
          [-7.40153218e-03, 6.03930248e-01, 1.01704308e-01],
          [-1.02960759e-02, 3.93481284e-03, -2.56594686e-03],
          [-2.65585735e-02, 3.10685217e-01, 5.90257974e-02],
          [-2.97909979e-02, 6.09658773e-01, 9.83101419e-02],
          [5.27935016e-03, -1.95547908e-01, -3.06803451e-02],
          [9.67095383e-03, -4.67827216e-01, -6.31183199e-02]],
    std=[[0.22265961, 0.19394593, 0.24823498],
         [0.14710804, 0.13572695, 0.16518279],
         [0.16562233, 0.12820609, 0.1770134],
         [0.25062919, 0.1896429, 0.24869254],
         [0.29278334, 0.29575863, 0.28972444],
         [0.16916984, 0.13424898, 0.17943313],
         [0.24760463, 0.18768265, 0.24697394],
         [0.28709979, 0.28541425, 0.29065647],
         [0.08867271, 0.02868353, 0.08192097],
         [0.21473598, 0.23872363, 0.22448061],
         [0.26021136, 0.3188117, 0.29020494],
         [0.08867271, 0.02868353, 0.08192097],
         [0.20729183, 0.2332424, 0.22969608],
         [0.26214967, 0.3125435, 0.29601641],
         [0.07129179, 0.06720073, 0.0811808],
         [0.17489889, 0.15827879, 0.19465977]])

# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[991.90641651, 862.69810047], [1012.08511619, 957.61720198],
          [1014.49360896, 974.59889655], [1015.67993223, 1055.61969227],
          [1012.53566238, 1082.80581721], [1009.22188073, 973.93984209],
          [1005.0694331, 1058.35166276], [1003.49327495, 1089.75631017],
          [1010.54615457, 1141.46165082], [1003.63254875, 1283.37687485],
          [1001.97780897, 1418.03079034], [1006.61419313, 1145.20131053],
          [999.60794074, 1287.13556333], [998.33830821, 1422.30463081],
          [1008.58017385, 1143.33148068], [1010.97561846, 1053.38953748],
          [1012.06704779, 925.75338048]],
    std=[[23374.39708662, 7213.93351296], [533.82975336, 219.70387631],
         [539.03326985, 218.9370412], [566.57219249, 233.32613405],
         [590.4265317, 269.2245025], [539.92993936, 218.53166338],
         [546.30605944, 228.43631598], [564.88616584, 267.85235566],
         [515.76216052, 206.72322146], [500.6260933, 223.24233285],
         [505.35940904, 268.4394148], [512.43406541, 202.93095363],
         [502.41443672, 218.70111819], [509.76363747, 267.67317375],
         [511.65693552, 204.13307947], [521.66823785, 205.96774166],
         [541.47940161, 226.01738951]])

train_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=14,
        root_name='root_position',
        remove_root=True),
    dict(
        type='NormalizeJointCoordinate',
        item='target',
        mean=joint_3d_normalize_param['mean'],
        std=joint_3d_normalize_param['std']),
    dict(
        type='NormalizeJointCoordinate',
        item='input_2d',
        mean=joint_2d_normalize_param['mean'],
        std=joint_2d_normalize_param['std']),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=[
            'target_image_path', 'flip_pairs', 'root_position',
            'root_position_index', 'target_mean', 'target_std'
        ])
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='Body3DMpiInf3dhpDataset',
        ann_file=f'{data_root}/annotations/mpi_inf_3dhp_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=train_data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Body3DMpiInf3dhpDataset',
        ann_file=f'{data_root}/annotations/mpi_inf_3dhp_test_valid.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=test_data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DMpiInf3dhpDataset',
        ann_file=f'{data_root}/annotations/mpi_inf_3dhp_test_valid.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=test_data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
