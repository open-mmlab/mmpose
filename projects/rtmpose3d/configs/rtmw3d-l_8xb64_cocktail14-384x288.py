_base_ = ['mmpose::_base_/default_runtime.py']

custom_imports = dict(imports=['rtmpose3d'], allow_failed_imports=False)

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
max_epochs = 270
stage2_num_epochs = 10
base_lr = 5e-4
num_keypoints = 133

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=2024)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=4096)

# codec settings
codec = dict(
    type='SimCC3DLabel',
    input_size=(288, 384, 288),
    sigma=(6., 6.93, 6.),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    root_index=(11, 12))

backbone_path = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth'  # noqa

# model settings
model = dict(
    type='TopdownPoseEstimator3D',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=backbone_path)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    head=dict(
        type='RTMW3DHead',
        in_channels=1024,
        out_channels=133,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.1,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=[
            dict(
                type='KLDiscretLossWithWeight',
                use_target_weight=True,
                beta=10.,
                label_softmax=True),
            dict(
                type='BoneLoss',
                joint_parents=[
                    0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30,
                    31, 32, 33, 34, 35, 36, 37, 38, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                    3, 50, 50, 51, 52, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3,
                    3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 7, 91, 92, 93, 94, 91, 96, 97, 98, 91, 100,
                    101, 102, 91, 104, 105, 106, 91, 108, 109, 110, 8, 112,
                    113, 114, 113, 112, 117, 118, 117, 112, 121, 122, 123, 112,
                    125, 126, 127, 112, 129, 130, 131
                ],
                use_target_weight=True,
                loss_weight=2.0)
        ],
        decoder=codec),
    # test_cfg=dict(flip_test=False, mode='2d')
    test_cfg=dict(flip_test=False))

# base dataset settings
data_mode = 'topdown'
dataset_type = 'H36MWholeBodyDataset'
backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='YOLOXHSVRandomAug'),
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
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='GenerateTarget', encoder=codec),
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
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# mapping

aic_coco133 = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12),
               (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)]

crowdpose_coco133 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                     (7, 12), (8, 13), (9, 14), (10, 15), (11, 16)]

mpii_coco133 = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
]

jhmdb_coco133 = [
    (3, 6),
    (4, 5),
    (5, 12),
    (6, 11),
    (7, 8),
    (8, 7),
    (9, 14),
    (10, 13),
    (11, 10),
    (12, 9),
    (13, 16),
    (14, 15),
]

halpe_coco133 = [(i, i)
                 for i in range(17)] + [(20, 17), (21, 20), (22, 18), (23, 21),
                                        (24, 19),
                                        (25, 22)] + [(i, i - 3)
                                                     for i in range(26, 136)]

posetrack_coco133 = [
    (0, 0),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
]

humanart_coco133 = [(i, i) for i in range(17)] + [(17, 99), (18, 120),
                                                  (19, 17), (20, 20)]

data_mode = 'topdown'
data_root = 'data/'

# train datasets
dataset_coco = dict(
    type='CocoWholeBodyDataset',
    data_root='data/coco/',
    data_mode='topdown',
    ann_file='annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='train2017/'),
    pipeline=[],
)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(img='pose/ai_challenge/ai_challenger_keypoint'
                     '_train_20170902/keypoint_train_images_20170902/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=aic_coco133)
    ],
)

dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=crowdpose_coco133)
    ],
)

dataset_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='pose/MPI/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=mpii_coco133)
    ],
)

dataset_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/Sub1_train.json',
    data_prefix=dict(img='pose/JHMDB/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=jhmdb_coco133)
    ],
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train_v1.json',
    data_prefix=dict(img='pose/Halpe/hico_20160224_det/images/train2015'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe_coco133)
    ],
)

dataset_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack18/annotations/posetrack18_train.json',
    data_prefix=dict(img='pose/PoseChallenge2018/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=posetrack_coco133)
    ],
)

dataset_humanart = dict(
    type='HumanArt21Dataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='HumanArt/annotations/training_humanart.json',
    filter_cfg=dict(scenes=['real_human']),
    data_prefix=dict(img='pose/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=humanart_coco133)
    ])

face_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[1.5, 2.0],
        rotate_factor=0),
]

wflw_coco133 = [(i * 2, 23 + i)
                for i in range(17)] + [(33 + i, 40 + i) for i in range(5)] + [
                    (42 + i, 45 + i) for i in range(5)
                ] + [(51 + i, 50 + i)
                     for i in range(9)] + [(60, 59), (61, 60), (63, 61),
                                           (64, 62), (65, 63), (67, 64),
                                           (68, 65), (69, 66), (71, 67),
                                           (72, 68), (73, 69),
                                           (75, 70)] + [(76 + i, 71 + i)
                                                        for i in range(20)]
dataset_wflw = dict(
    type='WFLWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='wflw/annotations/face_landmarks_wflw_train.json',
    data_prefix=dict(img='pose/WFLW/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=wflw_coco133), *face_pipeline
    ],
)

mapping_300w_coco133 = [(i, 23 + i) for i in range(68)]
dataset_300w = dict(
    type='Face300WDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='300w/annotations/face_landmarks_300w_train.json',
    data_prefix=dict(img='pose/300w/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=mapping_300w_coco133), *face_pipeline
    ],
)

cofw_coco133 = [(0, 40), (2, 44), (4, 42), (1, 49), (3, 45), (6, 47), (8, 59),
                (10, 62), (9, 68), (11, 65), (18, 54), (19, 58), (20, 53),
                (21, 56), (22, 71), (23, 77), (24, 74), (25, 85), (26, 89),
                (27, 80), (28, 31)]
dataset_cofw = dict(
    type='COFWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='cofw/annotations/cofw_train.json',
    data_prefix=dict(img='pose/COFW/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=cofw_coco133), *face_pipeline
    ],
)

lapa_coco133 = [(i * 2, 23 + i) for i in range(17)] + [
    (33 + i, 40 + i) for i in range(5)
] + [(42 + i, 45 + i) for i in range(5)] + [
    (51 + i, 50 + i) for i in range(4)
] + [(58 + i, 54 + i) for i in range(5)] + [(66, 59), (67, 60), (69, 61),
                                            (70, 62), (71, 63), (73, 64),
                                            (75, 65), (76, 66), (78, 67),
                                            (79, 68), (80, 69),
                                            (82, 70)] + [(84 + i, 71 + i)
                                                         for i in range(20)]
dataset_lapa = dict(
    type='LapaDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='LaPa/annotations/lapa_trainval.json',
    data_prefix=dict(img='pose/LaPa/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=lapa_coco133), *face_pipeline
    ],
)

dataset_wb = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='mmpose::_base_/datasets/coco_wholebody.py'),
    datasets=[dataset_coco, dataset_halpe],
    pipeline=[],
    test_mode=False,
)

dataset_body = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='mmpose::_base_/datasets/coco_wholebody.py'),
    datasets=[
        dataset_aic,
        dataset_crowdpose,
        dataset_mpii,
        dataset_jhmdb,
        dataset_posetrack,
        # dataset_humanart,
    ],
    pipeline=[],
    test_mode=False,
)

dataset_face = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='mmpose::_base_/datasets/coco_wholebody.py'),
    datasets=[
        dataset_wflw,
        dataset_300w,
        dataset_cofw,
        dataset_lapa,
    ],
    pipeline=[],
    test_mode=False,
)

hand_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[1.5, 2.0],
        rotate_factor=0),
]

interhand_left = [(21, 95), (22, 94), (23, 93), (24, 92), (25, 99), (26, 98),
                  (27, 97), (28, 96), (29, 103), (30, 102), (31, 101),
                  (32, 100), (33, 107), (34, 106), (35, 105), (36, 104),
                  (37, 111), (38, 110), (39, 109), (40, 108), (41, 91)]
interhand_right = [(i - 21, j + 21) for i, j in interhand_left]
interhand_coco133 = interhand_right + interhand_left

dataset_interhand2d = dict(
    type='InterHand2DDoubleDataset',
    data_root='data/interhand2.6m/',
    data_mode='topdown',
    ann_file='annotations/all/InterHand2.6M_train_data.json',
    camera_param_file='annotations/all/InterHand2.6M_train_camera.json',
    joint_file='annotations/all/InterHand2.6M_train_joint_3d.json',
    data_prefix=dict(img='images/train/'),
    sample_interval=10,
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=interhand_coco133,
        ), *hand_pipeline
    ],
)

dataset_interhand3d = dict(
    type='InterHand3DDataset',
    data_root='data/interhand2.6m/',
    data_mode='topdown',
    ann_file='annotations/all/InterHand2.6M_train_data.json',
    camera_param_file='annotations/all/InterHand2.6M_train_camera.json',
    joint_file='annotations/all/InterHand2.6M_train_joint_3d.json',
    use_gt_root_depth=True,
    rootnet_result_file=None,
    data_prefix=dict(img='images/train/'),
    sample_interval=10,
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=interhand_coco133,
        ), *hand_pipeline
    ],
)

dataset_hand = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='mmpose::_base_/datasets/coco_wholebody.py'),
    datasets=[dataset_interhand3d],
    pipeline=[],
    test_mode=False,
)

# ubody dataset
scenes = [
    'Magic_show', 'Entertainment', 'ConductMusic', 'Online_class', 'TalkShow',
    'Speech', 'Fitness', 'Interview', 'Olympic', 'TVShow', 'Singing',
    'SignLanguage', 'Movie', 'LiveVlog', 'VideoConference'
]
ubody_datasets = []
for scene in scenes:
    train_ann = f'annotations/{scene}/train_3dkeypoint_annotation.json'
    ubody = dict(
        type='UBody3dDataset',
        data_root='data/UBody/',
        ann_file=train_ann,
        data_mode='topdown',
        causal=True,
        seq_len=1,
        data_prefix=dict(img='images/'),
        subset_frac=0.1,
        pipeline=[])
    ubody_datasets.append(ubody)

# h3wb dataset
h3wb_dataset = dict(
    type='H36MWholeBodyDataset',
    ann_file='annotation_body3d/h3wb_train_bbox.npz',
    seq_len=1,
    causal=True,
    data_root='data/h36m/',
    data_prefix=dict(img='images/'),
    test_mode=False,
    pipeline=[])

train_datasets = [
    dataset_wb,
    dataset_body,
    dataset_face,
    *ubody_datasets,
    h3wb_dataset,
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        datasets=train_datasets,
        pipeline=train_pipeline,
        metainfo=dict(from_file='mmpose::_base_/datasets/h3wb.py'),
        test_mode=False))
# hooks
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

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1))

# eval h3wb
val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='H36MWholeBodyDataset',
        ann_file='annotation_body3d/h3wb_train_bbox.npz',
        seq_len=1,
        causal=True,
        data_root='data/h36m/',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='SimpleMPJPE', mode='mpjpe'),
    dict(type='SimpleMPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator

# eval coco
# val_dataloader = dict(
#     batch_size=64,
#     num_workers=10,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
#     dataset=dict(
#         type='CocoWholeBodyDataset',
#         data_root='data/coco/',
#         data_mode='topdown',
#         ann_file='annotations/coco_wholebody_val_v1.0.json',
#         data_prefix=dict(img='val2017/'),
#         test_mode=True,
#         bbox_file='data/coco/person_detection_results/'
#         'COCO_val2017_detections_AP_H_56_person.json',
#         pipeline=val_pipeline,
#     ))
# test_dataloader = val_dataloader

# # evaluators
# val_evaluator = dict(
#     type='CocoWholeBodyMetric',
#     ann_file='data/coco/' + 'annotations/coco_wholebody_val_v1.0.json')
# test_evaluator = val_evaluator

# hooks
# default_hooks = dict(
#     checkpoint=dict(
#         save_best='coco-wholebody/AP', rule='greater', max_keep_ckpts=1))
