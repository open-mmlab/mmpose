# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from mmpose.registry import MODELS


def resize(input: torch.Tensor,
           size: Optional[Union[Tuple[int, int], torch.Size]] = None,
           scale_factor: Optional[float] = None,
           mode: str = 'nearest',
           align_corners: Optional[bool] = None,
           warning: bool = True) -> torch.Tensor:
    """Resize a given input tensor using specified size or scale_factor.

    Args:
        input (torch.Tensor): The input tensor to be resized.
        size (Optional[Union[Tuple[int, int], torch.Size]]): The desired
            output size. Defaults to None.
        scale_factor (Optional[float]): The scaling factor for resizing.
            Defaults to None.
        mode (str): The interpolation mode. Defaults to 'nearest'.
        align_corners (Optional[bool]): Determines whether to align the
            corners when using certain interpolation modes. Defaults to None.
        warning (bool): Whether to display a warning when the input and
            output sizes are not ideal for alignment. Defaults to True.

    Returns:
        torch.Tensor: The resized tensor.
    """
    # Check if a warning should be displayed regarding input and output sizes
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would be more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')

    # Convert torch.Size to tuple if necessary
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)

    # Perform the resizing operation
    return F.interpolate(input, size, scale_factor, mode, align_corners)


@MODELS.register_module()
class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are
    fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without
    which any other models than torchvision.models.resnet[18,34,50,101] produce
    nans.
    """

    def __init__(self, n, eps: int = 1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def inverse_sigmoid(x: Tensor, eps: float = 1e-3) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
