# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmpose.models.necks import GlobalAveragePooling


def test_gap():
    """Test GlobalAveragePooling neck."""
    gap = GlobalAveragePooling()

    with pytest.raises(TypeError):
        gap(1)

    x0_shape = (32, 1024, 4, 4)
    x1_shape = (32, 2048, 2, 2)
    x0 = _demo_inputs(x0_shape)
    x1 = _demo_inputs(x1_shape)

    y = gap(x0)
    assert y.shape == torch.Size([32, 1024])

    y = gap([x0, x1])
    assert y[0].shape == torch.Size([32, 1024])
    assert y[1].shape == torch.Size([32, 2048])

    y = gap((x0, x1))
    assert y[0].shape == torch.Size([32, 1024])
    assert y[1].shape == torch.Size([32, 2048])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    """
    imgs = np.random.random(input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs
