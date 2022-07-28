# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import ImgDataPreprocessor

from mmpose.registry import MODELS


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks."""
