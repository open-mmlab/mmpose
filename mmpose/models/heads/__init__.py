# Copyright (c) OpenMMLab. All rights reserved.
from .base_head import BaseHead
from .coord_cls_heads import RTMCCHead, SimCCHead
from .heatmap_heads import (AssociativeEmbeddingHead, CIDHead, CPMHead,
                            HeatmapHead, MSPNHead, ViPNASHead)
from .hybrid_heads import DEKRHead, VisPredictHead
from .regression_heads import (DSNTHead, IntegralRegressionHead,
                               RegressionHead, RLEHead, TemporalRegressionHead,
                               TrajectoryRegressionHead)

__all__ = [
    'BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'RegressionHead', 'IntegralRegressionHead', 'SimCCHead', 'RLEHead',
    'DSNTHead', 'AssociativeEmbeddingHead', 'DEKRHead', 'VisPredictHead',
    'CIDHead', 'RTMCCHead', 'TemporalRegressionHead',
    'TrajectoryRegressionHead'
]
