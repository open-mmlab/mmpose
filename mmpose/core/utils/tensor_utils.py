# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.utils import is_seq_of
from torch import Tensor


def to_numpy(x: Union[Tensor, Sequence[Tensor]],
             return_device: bool = False,
             unzip: bool = False) -> Union[np.ndarray, tuple]:
    """Convert torch tensor to numpy.ndarray.

    Args:
        x (Tensor | Sequence[Tensor]): A single tensor or a sequence of
            tensors
        return_device (bool): Whether return the tensor device. Defaults to
            ``False``
        unzip (bool): Whether unzip the input sequence. Ddfaults to ``False``

    Returns:
        np.ndarray | tuple: If ``return_device`` is ``True``, return a tuple
        of converted numpy array(s) and the device indicator; otherwise only
        return the numpy array(s)
    """

    if isinstance(x, Tensor):
        arrays = x.detach().cpu().numpy()
        device = x.device
    elif is_seq_of(x, Tensor):
        if unzip:
            # convert (A, B) -> [(A[0], B[0]), (A[1], B[1]), ...]
            arrays = [
                tuple(to_numpy(_x[None, :]) for _x in _each)
                for _each in zip(*x)
            ]
        else:
            arrays = [to_numpy(_x) for _x in x]

        device = x[0].device

    else:
        raise ValueError(f'Invalid input type {type(x)}')

    if return_device:
        return arrays, device
    else:
        return arrays


def to_tensor(x: Union[np.ndarray, Sequence[np.ndarray]],
              device: Optional[Any] = None) -> Union[Tensor, Sequence[Tensor]]:
    """Convert numpy.ndarray to torch tensor.

    Args:
        x (np.ndarray | Sequence[np.ndarray]): A single np.ndarray or a
            sequence of tensors
        tensor (Any, optional): The device indicator. Defaults to ``None``

    Returns:
        tuple:
        - Tensor | Sequence[Tensor]: The converted Tensor or Tensor sequence
    """
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device)
    elif is_seq_of(x, np.ndarray):
        return [to_tensor(_x, device=device) for _x in x]
    else:
        raise ValueError(f'Invalid input type {type(x)}')
