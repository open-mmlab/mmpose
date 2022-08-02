# Copyright (c) OpenMMLab. All rights reserved.
from .base_head import BaseHead
from .heatmap_heads import CPMHead, HeatmapHead, MSPNHead, ViPNASHead

__all__ = ['BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead']
