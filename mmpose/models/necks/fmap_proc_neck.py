# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.models.utils.ops import resize
from mmpose.registry import MODELS


@MODELS.register_module()
class FeatureMapProcessor(nn.Module):
    """A PyTorch module for selecting, concatenating, and rescaling feature
    maps.

    Args:
        select_index (Optional[Union[int, Tuple[int]]], optional): Index or
            indices of feature maps to select. Defaults to None, which means
            all feature maps are used.
        concat (bool, optional): Whether to concatenate the selected feature
            maps. Defaults to False.
        scale_factor (float, optional): The scaling factor to apply to the
            feature maps. Defaults to 1.0.
        apply_relu (bool, optional): Whether to apply ReLU on input feature
            maps. Defaults to False.
        align_corners (bool, optional): Whether to align corners when resizing
            the feature maps. Defaults to False.
    """

    def __init__(
        self,
        select_index: Optional[Union[int, Tuple[int]]] = None,
        concat: bool = False,
        scale_factor: float = 1.0,
        apply_relu: bool = False,
        align_corners: bool = False,
    ):
        super().__init__()

        if isinstance(select_index, int):
            select_index = (select_index, )
        self.select_index = select_index
        self.concat = concat

        assert (
            scale_factor > 0
        ), f'the argument `scale_factor` must be positive, ' \
           f'but got {scale_factor}'
        self.scale_factor = scale_factor
        self.apply_relu = apply_relu
        self.align_corners = align_corners

    def forward(self, inputs: Union[Tensor, Sequence[Tensor]]
                ) -> Union[Tensor, List[Tensor]]:

        if not isinstance(inputs, (tuple, list)):
            sequential_input = False
            inputs = [inputs]
        else:
            sequential_input = True

            if self.select_index is not None:
                inputs = [inputs[i] for i in self.select_index]

            if self.concat:
                inputs = self._concat(inputs)

        if self.apply_relu:
            inputs = [F.relu(x) for x in inputs]

        if self.scale_factor != 1.0:
            inputs = self._rescale(inputs)

        if not sequential_input:
            inputs = inputs[0]

        return inputs

    def _concat(self, inputs: Sequence[Tensor]) -> List[Tensor]:
        size = inputs[0].shape[-2:]
        resized_inputs = [
            resize(
                x,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        return [torch.cat(resized_inputs, dim=1)]

    def _rescale(self, inputs: Sequence[Tensor]) -> List[Tensor]:
        rescaled_inputs = [
            resize(
                x,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=self.align_corners,
            ) for x in inputs
        ]
        return rescaled_inputs
