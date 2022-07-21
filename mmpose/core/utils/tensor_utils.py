# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torch import Tensor


def _to_numpy(x: Tensor) -> np.ndarray:
    """Convert a torch tensor to numpy.ndarray.

    Args:
        x (Tensor): A torch tensor

    Returns:
        np.ndarray: The converted numpy array
    """
    return x.detach().cpu().numpy()
