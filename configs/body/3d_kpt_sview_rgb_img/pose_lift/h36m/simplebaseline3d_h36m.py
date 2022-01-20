_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h36m.py'
]
evaluation = dict(interval=10, metric=['mpjpe', 'p-mpjpe'], save_best='MPJPE')

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
data_root = 'data/h36m'
data_cfg = dict(
    num_joints=17,
    seq_len=1,
    seq_frame_interval=1,
    causal=True,
    joint_2d_src='gt',
    need_camera_param=False,
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

# 2D joint normalization parameters
# From file: '{data_root}/annotation_body3d/fps50/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[532.08351635, 419.74137558], [531.80953144, 418.2607141],
          [530.68456967, 493.54259285], [529.36968722, 575.96448516],
          [532.29767646, 421.28483336], [531.93946631, 494.72186795],
          [529.71984447, 578.96110365], [532.93699382, 370.65225054],
          [534.1101856, 317.90342311], [534.55416813, 304.24143901],
          [534.86955004, 282.31030885], [534.11308566, 330.11296796],
          [533.53637525, 376.2742511], [533.49380107, 391.72324565],
          [533.52579142, 330.09494668], [532.50804964, 374.190479],
          [532.72786934, 380.61615716]],
    std=[[107.73640054, 63.35908715], [119.00836213, 64.1215443],
         [119.12412107, 50.53806215], [120.61688045, 56.38444891],
         [101.95735275, 62.89636486], [106.24832897, 48.41178119],
         [108.46734966, 54.58177071], [109.07369806, 68.70443672],
         [111.20130351, 74.87287863], [111.63203838, 77.80542514],
         [113.22330788, 79.90670556], [105.7145833, 73.27049436],
         [107.05804267, 73.93175781], [107.97449418, 83.30391802],
         [121.60675105, 74.25691526], [134.34378973, 77.48125087],
         [131.79990652, 89.86721124]])

train_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
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
        type='Body3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Body3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
