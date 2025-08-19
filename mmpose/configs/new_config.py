# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from mmpose.configs._base_.default_runtime import *

from mmengine.dataset import DefaultSampler
from mmengine.model import PretrainedInit
from mmengine.optim import LinearLR, MultiStepLR
from torch.optim import Adam

from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import OneHand10KDataset
from mmpose.datasets import GenerateTarget
from mmpose.datasets import GetBBoxCenterScale
from mmpose.datasets import LoadImage
from mmpose.datasets import PackPoseInputs
from mmpose.datasets import RandomFlip
from mmpose.datasets import TopdownAffine
from mmpose.datasets.transforms.common_transforms import RandomBBoxTransform
from mmpose.evaluation import PCKAccuracy, AUC, EPE
from mmpose.models import HeatmapHead
from mmpose.models import HRNet
from mmpose.models import KeypointMSELoss
from mmpose.models import FeatureMapProcessor
from mmpose.models import PoseDataPreprocessor
from mmpose.models import TopdownPoseEstimator
#

# runtime
train_cfg.update(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type=Adam,
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(type=LinearLR, begin=0, end=500, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(
        type=MultiStepLR,
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks.update(checkpoint=dict(save_best='AUC', rule='greater'))

# codec settings
codec = dict(
    type=MSRAHeatmap,
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2,
    unbiased=True
)

# model settings
model = dict(
    type=TopdownPoseEstimator,
    data_preprocessor=dict(
        type=PoseDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type=HRNet,
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
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True),
            upsample=dict(mode='bilinear', align_corners=False)),
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint='open-mmlab://msra/hrnetv2_w18',
        )),
    neck=dict(
        type=FeatureMapProcessor,
        concat=True,
    ),
    head=dict(
        type=HeatmapHead,
        in_channels=270,
        out_channels=21,
        deconv_out_channels=None,
        conv_out_channels=(270, ),
        conv_kernel_sizes=(1, ),
        loss=dict(type=KeypointMSELoss, use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = OneHand10KDataset
data_mode = 'topdown'
data_root = 'data/pose/OneHand10K/'

# pipelines
train_pipeline = [
    dict(type=LoadImage),
    dict(type=GetBBoxCenterScale),
    dict(type=RandomFlip, direction='horizontal'),
    dict(
        type=RandomBBoxTransform, rotate_factor=180,
        scale_factor=(0.7, 1.3)),
    dict(type=TopdownAffine, input_size=codec['input_size']),
    dict(type=GenerateTarget, encoder=codec),
    dict(type=PackPoseInputs)
]
val_pipeline = [
    dict(type=LoadImage),
    dict(type=GetBBoxCenterScale),
    dict(type=TopdownAffine, input_size=codec['input_size']),
    dict(type=PackPoseInputs)
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/onehand10k_train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/onehand10k_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type=PCKAccuracy, thr=0.2),
    dict(type=AUC),
    dict(type=EPE),
]
test_evaluator = val_evaluator
