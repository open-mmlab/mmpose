# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .bbox_loss import IoULoss
from .classification_loss import (BCELoss, JSDiscretLoss, KLDiscretLoss,
                                  VariFocalLoss)
from .fea_dis_loss import FeaLoss
from .heatmap_loss import (AdaptiveWingLoss, KeypointMSELoss,
                           KeypointOHKMMSELoss, MLECCLoss)
from .logit_dis_loss import KDLoss
from .loss_wrappers import CombinedLoss, MultipleLossWrapper
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss,
                              MPJPEVelocityJointLoss, MSELoss, OKSLoss,
                              RLELoss, SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss)

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss',
    'MPJPEVelocityJointLoss', 'FeaLoss', 'KDLoss', 'OKSLoss', 'IoULoss',
    'VariFocalLoss', 'MLECCLoss'
]
