# Copyright (c) OpenMMLab. All rights reserved.
"""MMPose provides following registry nodes to support using modules across
projects.

Each node is a child of the root registry in MMEngine.
More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import Registry

# manage data-related modules
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS)
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS)

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS)

# manage visualizer
VISUALIZERS = Registry('visualizer', parent=MMENGINE_VISUALIZERS)
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=MMENGINE_VISBACKENDS)

# manage all kinds of metrics
METRICS = Registry('metric', parent=MMENGINE_METRICS)

# manager keypoint encoder/decoder
KEYPOINT_CODECS = Registry('KEYPOINT_CODECS')

# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook', parent=MMENGINE_HOOKS)
