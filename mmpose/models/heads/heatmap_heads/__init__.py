# Copyright (c) OpenMMLab. All rights reserved.
from .heatmap_head import HeatmapHead
from .multi_stage_heatmap_head import (MultiStageHeatmapHead,
                                       MultiStageMultiUnitHeatmapHead)

__all__ = [
    'HeatmapHead', 'MultiStageHeatmapHead', 'MultiStageMultiUnitHeatmapHead'
]
