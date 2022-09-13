# Copyright (c) OpenMMLab. All rights reserved.
from .associative_embedding import AssociativeEmbedding
from .integral_regression_label import IntegralRegressionLabel
from .megvii_heatmap import MegviiHeatmap
from .msra_heatmap import MSRAHeatmap
from .regression_label import RegressionLabel
from .simcc_label import SimCCLabel
from .udp_heatmap import UDPHeatmap

__all__ = [
    'MSRAHeatmap', 'MegviiHeatmap', 'UDPHeatmap', 'RegressionLabel',
    'SimCCLabel', 'IntegralRegressionLabel', 'AssociativeEmbedding'
]
