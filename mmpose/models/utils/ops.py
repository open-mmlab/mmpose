# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple, Union

import torch
from torch.nn import functional as F


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
