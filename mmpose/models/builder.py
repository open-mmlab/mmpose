# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmpose.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
POSE_ESTIMATORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_pose_estimator(cfg):
    """Build pose estimator."""
    return POSE_ESTIMATORS.build(cfg)


def build_posenet(cfg):
    """Build posenet."""
    warnings.warn(
        '``build_posenet`` will be deprecated soon, '
        'please use ``build_pose_estimator`` instead.', DeprecationWarning)
    return build_pose_estimator(cfg)
