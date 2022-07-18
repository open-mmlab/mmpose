# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmpose.core import WeightNormClipHook


def test_weight_norm_clip():
    torch.manual_seed(0)

    module = torch.nn.Linear(2, 2, bias=False)
    module.weight.data.fill_(2)
    WeightNormClipHook(max_norm=1.0).register(module)

    x = torch.rand(1, 2).requires_grad_()
    _ = module(x)

    weight_norm = module.weight.norm().item()
    np.testing.assert_almost_equal(weight_norm, 1.0, decimal=6)
