# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss
from .heatmap_loss import (AdaptiveWingLoss, KeypointMSELoss,
                           KeypointOHKMMSELoss)
from .loss_wrappers import CombinedLoss, MultipleLoss, MultipleLossWrapper
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss, SoftWingLoss,
                              WingLoss)

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'SmoothL1Loss', 'WingLoss', 'MPJPELoss', 'MSELoss',
    'L1Loss', 'BCELoss', 'BoneLoss', 'SemiSupervisionLoss', 'SoftWingLoss',
    'AdaptiveWingLoss', 'RLELoss', 'KLDiscretLoss', 'MultipleLossWrapper',
    'JSDiscretLoss', 'CombinedLoss', 'MultipleLoss', 'AssociativeEmbeddingLoss'
]
