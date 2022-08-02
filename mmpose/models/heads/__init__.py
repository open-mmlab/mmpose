# Copyright (c) OpenMMLab. All rights reserved.
from .base_head import BaseHead
from .heatmap_heads import (HeatmapHead, SimCCHead, CPMHead,
                            HeatmapHead, MSPNHead, ViPNASHead)
from .regression_heads import (DSNTHead, IntegralRegressionHead,
                               RegressionHead, RLEHead)

__all__ = [
    'BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead', 
    'RegressionHead', 'IntegralRegressionHead', 'SimCCHead', 'RLEHead',
    'DSNTHead'
]

