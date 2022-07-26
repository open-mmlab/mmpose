# Copyright (c) OpenMMLab. All rights reserved.
from .base_head import BaseHead
from .heatmap_heads import CPMHead, HeatmapHead, MSPNHead, ViPNASHead
from .regression_heads import IntegralRegressionHead, RegressionHead

__all__ = [
    'BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead', 'RegressionHead', 'IntegralRegressionHead',
    'SimCCHead'
]
