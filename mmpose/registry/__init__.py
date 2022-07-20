# Copyright (c) OpenMMLab. All rights reserved.
from .registry import (DATASETS, KEYPOINT_CODECS, METRICS, MODELS, TRANSFORMS,
                       VISBACKENDS, VISUALIZERS)

__all__ = [
    'DATASETS', 'METRICS', 'TRANSFORMS', 'VISBACKENDS', 'VISUALIZERS',
    'MODELS', 'KEYPOINT_CODECS'
]
