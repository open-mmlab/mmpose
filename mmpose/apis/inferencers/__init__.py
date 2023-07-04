# Copyright (c) OpenMMLab. All rights reserved.
from .mmpose_inferencer import MMPoseInferencer
from .pose2d_inferencer import Pose2DInferencer
from .pose3d_inferencer import Pose3DInferencer
from .utils import get_model_aliases

__all__ = [
    'Pose2DInferencer', 'MMPoseInferencer', 'get_model_aliases',
    'Pose3DInferencer'
]
