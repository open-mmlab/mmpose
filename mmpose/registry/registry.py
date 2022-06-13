# Copyright (c) OpenMMLab. All rights reserved.
"""MMPose provides following registry nodes to support using modules across
projects. Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import Registry

# manage data-related modules
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS)

# manage all kinds of metrics
METRICS = Registry('metric', parent=MMENGINE_METRICS)
