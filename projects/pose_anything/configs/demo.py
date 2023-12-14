log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=20)
evaluation = dict(
    interval=25,
    metric=['PCK', 'NME', 'AUC', 'EPE'],
    key_indicator='PCK',
    gpu_collect=True,
    res_folder='')
optimizer = dict(
    type='Adam',
    lr=1e-5,
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[160, 180])
total_epochs = 200
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[
        [
            0,
        ],
    ],
    inference_channel=[
        0,
    ],
    max_kpt_num=100)

# model settings
model = dict(
    type='TransformerPoseTwoStage',
    pretrained='swinv2_large',
    encoder_config=dict(
        type='SwinTransformerV2',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=16,
        pretrained_window_sizes=[12, 12, 12, 6],
        drop_path_rate=0.2,
        img_size=256,
    ),
    keypoint_head=dict(
        type='TwoStageHead',
        in_channels=1536,
        transformer=dict(
            type='TwoStageSupportRefineTransformer',
            d_model=384,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=1536,
            dropout=0.1,
            similarity_proj_dim=384,
            dynamic_proj_dim=192,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=True),
        share_kpt_branch=False,
        num_decoder_layer=3,
        with_heatmap_loss=True,
        support_pos_embed=False,
        heatmap_loss_weight=2.0,
        skeleton_loss_weight=0.02,
        num_samples=0,
        support_embedding_type='fixed',
        num_support=100,
        support_order_dropout=-1,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=192, normalize=True)),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15,
        scale_factor=0.15),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file',
            'joints_3d',
            'joints_3d_visible',
            'center',
            'scale',
            'rotation',
            'bbox_score',
            'flip_pairs',
            'category_id',
            'skeleton',
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=1),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file',
            'joints_3d',
            'joints_3d_visible',
            'center',
            'scale',
            'rotation',
            'bbox_score',
            'flip_pairs',
            'category_id',
            'skeleton',
        ]),
]

test_pipeline = valid_pipeline

data_root = 'data/mp100'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/mp100_all.json',
        img_prefix=f'{data_root}/images/',
        # img_prefix=f'{data_root}',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        pipeline=train_pipeline),
    val=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_val.json',
        img_prefix=f'{data_root}/images/',
        # img_prefix=f'{data_root}',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        num_queries=15,
        num_episodes=100,
        pipeline=valid_pipeline),
    test=dict(
        type='TestPoseDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_test.json',
        img_prefix=f'{data_root}/images/',
        # img_prefix=f'{data_root}',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        num_queries=15,
        num_episodes=200,
        pck_threshold_list=[0.05, 0.10, 0.15, 0.2, 0.25],
        pipeline=test_pipeline),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

shuffle_cfg = dict(interval=1)
