# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.config import ConfigDict

from mmpose.core.data_structures.pose_data_sample import PoseDataSample

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

SampleList = List[PoseDataSample]
OptSampleList = Optional[SampleList]

ForwardResults = Union[Dict[str, torch.Tensor], List[PoseDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
