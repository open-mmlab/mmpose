# Copyright (c) OpenMMLab. All rights reserved.
from .dsnt_head import DSNTHead
from .integral_regression_head import IntegralRegressionHead
from .regression_head import RegressionHead
from .rle_head import RLEHead
from .temporal_regression_head import TemporalRegressionHead
from .trajectory_regression_head import TrajectoryRegressionHead

__all__ = [
    'RegressionHead',
    'IntegralRegressionHead',
    'DSNTHead',
    'RLEHead',
    'TemporalRegressionHead',
    'TrajectoryRegressionHead',
]
