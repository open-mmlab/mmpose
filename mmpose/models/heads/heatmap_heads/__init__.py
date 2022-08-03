# Copyright (c) OpenMMLab. All rights reserved.
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead
from .mspn_head import MSPNHead
from .simcc_head import SimCCHead
from .vipnas_head import ViPNASHead

__all__ = ['HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead', 'SimCCHead']
