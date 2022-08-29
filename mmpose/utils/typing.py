# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmpose.structures import PoseDataSample

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]
# Type hint of data samples
SampleList = List[PoseDataSample]
OptSampleList = Optional[SampleList]
InstanceList = List[InstanceData]
PixelDataList = List[PixelData]
Predictions = Union[InstanceList, Tuple[InstanceList, PixelDataList]]
# Type hint of model outputs
ForwardResults = Union[Dict[str, Tensor], List[PoseDataSample], Tuple[Tensor],
                       Tensor]
# Type hint of features
#   - Tuple[Tensor]: multi-level features extracted by the network
#   - List[Tuple[Tensor]]: multiple feature pyramids for TTA
Features = Union[Tuple[Tensor], List[Tuple[Tensor]]]
