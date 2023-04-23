_base_ = ['mmpose::_base_/default_runtime.py']

# 数据集类型及路径
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/ape/'

# 三角板关键点检测数据集-元数据
dataset_info = dict(from_file='configs/_base_/datasets/linemod.py')

# NUM_KEYPOINTS = len(dataset_info['keypoint_info'])
NUM_KEYPOINTS = 8

# 训练超参数
max_epochs = 300 # 训练 epoch 总数
val_interval = 10 # 每隔多少个 epoch 保存一次权重文件
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 32
val_batch_size = 8
stage2_num_epochs = 0
base_lr = 5e-4
randomness = dict(seed=21)

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 学习率
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=20),
    dict(
        # use cosine lr from 210 to 420 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(12, 12),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# 不同模型的 config： https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose/rtmpose/body_2d_keypoint

## 模型：RTMPose-S
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/' +
            'rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth',  # noqa E501
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=NUM_KEYPOINTS,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

## 模型：RTMPose-M
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True),
#     backbone=dict(
#         _scope_='mmdet',
#         type='CSPNeXt',
#         arch='P5',
#         expand_ratio=0.5,
#         deepen_factor=0.67,
#         widen_factor=0.75,
#         out_indices=(4, ),
#         channel_attention=True,
#         norm_cfg=dict(type='SyncBN'),
#         act_cfg=dict(type='SiLU'),
#         init_cfg=dict(
#             type='Pretrained',
#             prefix='backbone.',
#             checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
#             'rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
#         )),
#     head=dict(
#         type='RTMCCHead',
#         in_channels=768,
#         out_channels=NUM_KEYPOINTS,
#         input_size=codec['input_size'],
#         in_featuremap_size=(6, 8),
#         simcc_split_ratio=codec['simcc_split_ratio'],
#         final_layer_kernel_size=7,
#         gau_cfg=dict(
#             hidden_dims=256,
#             s=128,
#             expansion_factor=2,
#             dropout_rate=0.,
#             drop_path=0.,
#             act_fn='SiLU',
#             use_rel_bias=False,
#             pos_enc=False),
#         loss=dict(
#             type='KLDiscretLoss',
#             use_target_weight=True,
#             beta=10.,
#             label_softmax=True),
#         decoder=codec),
#     test_cfg=dict(flip_test=True))

## 模型：RTMPose-L
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True),
#     backbone=dict(
#         _scope_='mmdet',
#         type='CSPNeXt',
#         arch='P5',
#         expand_ratio=0.5,
#         deepen_factor=1.,
#         widen_factor=1.,
#         out_indices=(4, ),
#         channel_attention=True,
#         norm_cfg=dict(type='SyncBN'),
#         act_cfg=dict(type='SiLU'),
#         init_cfg=dict(
#             type='Pretrained',
#             prefix='backbone.',
#             checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
#             'rtmpose/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'  # noqa
#         )),
#     head=dict(
#         type='RTMCCHead',
#         in_channels=1024,
#         out_channels=NUM_KEYPOINTS,
#         input_size=codec['input_size'],
#         in_featuremap_size=(6, 8),
#         simcc_split_ratio=codec['simcc_split_ratio'],
#         final_layer_kernel_size=7,
#         gau_cfg=dict(
#             hidden_dims=256,
#             s=128,
#             expansion_factor=2,
#             dropout_rate=0.,
#             drop_path=0.,
#             act_fn='SiLU',
#             use_rel_bias=False,
#             pos_enc=False),
#         loss=dict(
#             type='KLDiscretLoss',
#             use_target_weight=True,
#             beta=10.,
#             label_softmax=True),
#         decoder=codec),
#     test_cfg=dict(flip_test=True))


backend_args = dict(backend='local')
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/',
#         f'{data_root}': 's3://openmmlab/datasets/detection/coco/'
#     }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=30),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='ChannelShuffle', p=0.5),
            dict(type='CLAHE', p=0.5),
            # dict(type='Downscale', scale_min=0.7, scale_max=0.9, p=0.2),
            dict(type='ColorJitter', p=0.5),
            dict(
                type='CoarseDropout',
                max_holes=4,
                max_height=0.3,
                max_width=0.3,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
# train_pipeline = [
#     dict(type='LoadImage', backend_args=backend_args),
#     dict(type='GetBBoxCenterScale'),
#     dict(type='TopdownAffine', input_size=codec['input_size']),
#     dict(type='GenerateTarget', encoder=codec),
#     dict(type='PackPoseInputs')
# ]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='linemod_preprocessed_train.json',
        data_prefix=dict(img='rgb/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    # dataset=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     metainfo=dataset_info,
    #     data_mode=data_mode,
    #     ann_file='val_coco.json',
    #     # bbox_file=f'{data_root}person_detection_results/'
    #     # 'COCO_val2017_detections_AP_H_56_person.json',
    #     data_prefix=dict(img='images/'),
    #     test_mode=True,
    #     pipeline=val_pipeline,
    # )
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='linemod_preprocessed_test.json',
        data_prefix=dict(img='rgb/'),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
# default_hooks = dict(
#     checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1),
#     logger=dict(interval=1),
# )

# default_hooks = {
#     'checkpoint': {'save_best': 'coco/AP','rule': 'greater','max_keep_ckpts': 1},
#     'logger': {'interval': 1}
# }

default_hooks = {
    'checkpoint': {'save_best': 'PCK','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
# val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'val_coco.json')
# val_evaluator = dict(type='PCKAccuracy')
# val_evaluator = [dict(type='CocoMetric', ann_file=data_root + 'val_coco.json'), dict(type='PCKAccuracy')]

# val_evaluator = [
#     dict(type='CocoMetric', ann_file=data_root + 'val_coco.json'),
#     dict(type='PCKAccuracy'),
#     dict(type='AUC'),
#     dict(type='NME', norm_mode='keypoint_distance')
# ]

val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'linemod_preprocessed_test.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[1, 2])
]

test_evaluator = val_evaluator


