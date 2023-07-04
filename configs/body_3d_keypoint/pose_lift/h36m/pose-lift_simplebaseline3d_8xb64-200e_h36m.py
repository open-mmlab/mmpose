_base_ = ['../../../_base_/default_runtime.py']

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=200, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))

# learning policy
param_scheduler = [
    dict(type='StepLR', step_size=100000, gamma=0.96, end=80, by_epoch=False)
]

auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1))

# codec settings
# 3D keypoint normalization parameters
# From file: '{data_root}/annotation_body3d/fps50/joint3d_rel_stats.pkl'
target_mean = [[-2.55652589e-04, -7.11960570e-03, -9.81433052e-04],
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
target_std = [[0.11072244, 0.02238818, 0.07246294],
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
              [0.24400695, 0.23975028, 0.25520584]]
# 2D keypoint normalization parameters
# From file: '{data_root}/annotation_body3d/fps50/joint2d_stats.pkl'
keypoints_mean = [[532.08351635, 419.74137558], [531.80953144, 418.2607141],
                  [530.68456967, 493.54259285], [529.36968722, 575.96448516],
                  [532.29767646, 421.28483336], [531.93946631, 494.72186795],
                  [529.71984447, 578.96110365], [532.93699382, 370.65225054],
                  [534.1101856, 317.90342311], [534.55416813, 304.24143901],
                  [534.86955004, 282.31030885], [534.11308566, 330.11296796],
                  [533.53637525, 376.2742511], [533.49380107, 391.72324565],
                  [533.52579142, 330.09494668], [532.50804964, 374.190479],
                  [532.72786934, 380.61615716]],
keypoints_std = [[107.73640054, 63.35908715], [119.00836213, 64.1215443],
                 [119.12412107, 50.53806215], [120.61688045, 56.38444891],
                 [101.95735275, 62.89636486], [106.24832897, 48.41178119],
                 [108.46734966, 54.58177071], [109.07369806, 68.70443672],
                 [111.20130351, 74.87287863], [111.63203838, 77.80542514],
                 [113.22330788, 79.90670556], [105.7145833, 73.27049436],
                 [107.05804267, 73.93175781], [107.97449418, 83.30391802],
                 [121.60675105, 74.25691526], [134.34378973, 77.48125087],
                 [131.79990652, 89.86721124]]
codec = dict(
    type='ImagePoseLifting',
    num_keypoints=17,
    root_index=0,
    remove_root=True,
    target_mean=target_mean,
    target_std=target_std,
    keypoints_mean=keypoints_mean,
    keypoints_std=keypoints_std)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='TCN',
        in_channels=2 * 17,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(1, 1, 1),
        dropout=0.5,
    ),
    head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=16,
        loss=dict(type='MSELoss'),
        decoder=codec,
    ))

# base dataset settings
dataset_type = 'Human36mDataset'
data_root = 'data/h36m/'

# pipelines
train_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root', 'target_root_index', 'target_mean',
                   'target_std'))
]
val_pipeline = train_pipeline

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_train.npz',
        seq_len=1,
        causal=True,
        keypoint_2d_src='gt',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        seq_len=1,
        causal=True,
        keypoint_2d_src='gt',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='MPJPE', mode='mpjpe'),
    dict(type='MPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator
