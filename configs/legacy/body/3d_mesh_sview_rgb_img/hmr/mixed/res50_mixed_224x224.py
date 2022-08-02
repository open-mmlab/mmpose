_base_ = ['../../../../_base_/default_runtime.py']
use_adversarial_train = True

optimizer = dict(
    generator=dict(type='Adam', lr=2.5e-4),
    discriminator=dict(type='Adam', lr=1e-4))

optimizer_config = None

lr_config = dict(policy='Fixed', by_epoch=False)

total_epochs = 100
img_res = 224

# model settings
model = dict(
    type='ParametricMesh',
    pretrained=None,
    backbone=dict(type='ResNet', depth=50),
    mesh_head=dict(
        type='HMRMeshHead',
        in_channels=2048,
        smpl_mean_params='models/smpl/smpl_mean_params.npz',
    ),
    disc=dict(),
    smpl=dict(
        type='SMPL',
        smpl_path='models/smpl',
        joints_regressor='models/smpl/joints_regressor_cmr.npy'),
    train_cfg=dict(disc_step=1),
    test_cfg=dict(),
    loss_mesh=dict(
        type='MeshLoss',
        joints_2d_loss_weight=100,
        joints_3d_loss_weight=1000,
        vertex_loss_weight=20,
        smpl_pose_loss_weight=30,
        smpl_beta_loss_weight=0.2,
        focal_length=5000,
        img_res=img_res),
    loss_gan=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1))

data_cfg = dict(
    image_size=[img_res, img_res],
    iuv_size=[img_res // 4, img_res // 4],
    num_joints=24,
    use_IUV=False,
    uv_type='BF')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MeshRandomChannelNoise', noise_factor=0.4),
    dict(type='MeshRandomFlip', flip_prob=0.5),
    dict(type='MeshGetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img', 'joints_2d', 'joints_2d_visible', 'joints_3d',
            'joints_3d_visible', 'pose', 'beta', 'has_smpl'
        ],
        meta_keys=['image_file', 'center', 'scale', 'rotation']),
]

train_adv_pipeline = [dict(type='Collect', keys=['mosh_theta'], meta_keys=[])]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MeshAffine'),
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
        meta_keys=['image_file', 'center', 'scale', 'rotation']),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='MeshAdversarialDataset',
        train_dataset=dict(
            type='MeshMixDataset',
            configs=[
                dict(
                    ann_file='data/mesh_annotation_files/h36m_train.npz',
                    img_prefix='data/h36m_train',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/'
                    'mpi_inf_3dhp_train.npz',
                    img_prefix='data/mpi_inf_3dhp',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/'
                    'lsp_dataset_original_train.npz',
                    img_prefix='data/lsp_dataset_original',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/hr-lspet_train.npz',
                    img_prefix='data/hr-lspet',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/mpii_train.npz',
                    img_prefix='data/mpii',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/coco_2014_train.npz',
                    img_prefix='data/coco',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline)
            ],
            partition=[0.35, 0.15, 0.1, 0.10, 0.10, 0.2]),
        adversarial_dataset=dict(
            type='MoshDataset',
            ann_file='data/mesh_annotation_files/CMU_mosh.npz',
            pipeline=train_adv_pipeline),
    ),
    test=dict(
        type='MeshH36MDataset',
        ann_file='data/mesh_annotation_files/h36m_valid_protocol2.npz',
        img_prefix='data/Human3.6M',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
    ),
)
