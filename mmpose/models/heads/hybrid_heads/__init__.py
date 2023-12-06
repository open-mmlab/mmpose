# Copyright (c) OpenMMLab. All rights reserved.
from .dekr_head import DEKRHead
from .onestage_rtmpose_head import OneStageRTMHead
from .vis_head import VisPredictHead
from .yoloxpose_head import YOLOXPoseHead

__all__ = ['DEKRHead', 'VisPredictHead', 'YOLOXPoseHead', 'OneStageRTMHead']
