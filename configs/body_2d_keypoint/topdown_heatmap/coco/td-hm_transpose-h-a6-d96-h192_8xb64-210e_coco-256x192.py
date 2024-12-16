_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-4,
    weight_decay=0.0001,
))

# learning policy
param_scheduler = [
    dict(
        # use cosine lr from 105 to 210 epoch
        type='CosineAnnealingLR',
        eta_min=0.00001,
        by_epoch=True,
        T_max=210,
        begin=0,
        end=210,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

default_hooks = {
    'checkpoint': {'save_best': 'coco/AP', 'rule': 'greater', 'max_keep_ckpts': 2}
}

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='TransPose',
        image_size=[192, 256],
        dim=96,
        dim_feedforward=192,
        encoder_layers_num=6,
        num_head=1,
        pos_embedding_type='sine',
        num_deconv_layers=None,#donv and layers are configuration parameters when type_cnn is resnet
        num_deconv_filters=None,
        num_deconv_kernels=None,
        deconv_with_bias=None,
        layers=None,
        final_conv_kernel=1,
        num_joints=17,
        out_indices=-1,
        frozen_stages=-1,
        type_cnn='hrnet',
        type_hrnet='w48',#type_hrnet is a configuration parameter when type_cnn is hrnet
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/yangsenius/TransPose/releases/'
                       'download/Hub/tp_h_48_256x192_enc6_d96_h192_mh1.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=17,
        out_channels=17,
        deconv_out_channels=[],
        deconv_kernel_sizes=[],
        final_layer = None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
                  'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator