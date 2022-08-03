# Copyright (c) OpenMMLab. All rights reserved.
from .megvii_heatmap import MegviiHeatmap
from .msra_heatmap import MSRAHeatmap
from .regression_label import RegressionLabel
from .simcc_label import SimCCLabel
from .udp_heatmap import UDPHeatmap
from .wrappers import MultiLevelHeatmapEncoder

__all__ = [
    'MSRAHeatmap', 'MegviiHeatmap', 'UDPHeatmap', 'RegressionLabel',
    'SimCCLabel', 'MultiLevelHeatmapEncoder'
]
