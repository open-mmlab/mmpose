# Copyright (c) OpenMMLab. All rights reserved.
from .ae_higher_resolution_head import AEHigherResolutionHead
from .ae_simple_head import AESimpleHead
from .deeppose_regression_head import DeepposeRegressionHead
from .hmr_head import HMRMeshHead
from .interhand_3d_head import Interhand3DHead
from .temporal_regression_head import TemporalRegressionHead
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from .topdown_heatmap_multi_stage_head import (TopdownHeatmapMSMUHead,
                                               TopdownHeatmapMultiStageHead)
from .topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from .vipnas_heatmap_simple_head import ViPNASHeatmapSimpleHead

__all__ = [
    'TopdownHeatmapSimpleHead', 'TopdownHeatmapMultiStageHead',
    'TopdownHeatmapMSMUHead', 'TopdownHeatmapBaseHead',
    'AEHigherResolutionHead', 'AESimpleHead', 'DeepposeRegressionHead',
    'TemporalRegressionHead', 'Interhand3DHead', 'HMRMeshHead',
    'ViPNASHeatmapSimpleHead'
]
