# Copyright (c) OpenMMLab. All rights reserved.
from .dsnt_head import DSNTHead
from .integral_regression_head import IntegralRegressionHead
from .regression_head import RegressionHead
from .rle_head import RLEHead
from .soft_argmax_head import SoftArgmaxHead

__all__ = [
    'RegressionHead', 'IntegralRegressionHead', 'DSNTHead', 'RLEHead',
    'SoftArgmaxHead'
]
