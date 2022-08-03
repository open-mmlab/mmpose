# Copyright (c) OpenMMLab. All rights reserved.
from .classification_loss import BCELoss, KLDiscretLoss
from .heatmap_loss import AdaptiveWingLoss
from .mesh_loss import GANLoss, MeshLoss
from .mse_loss import KeypointMSELoss, KeypointOHKMMSELoss
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss, SoftWingLoss,
                              WingLoss)

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'MeshLoss', 'GANLoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss'
]
