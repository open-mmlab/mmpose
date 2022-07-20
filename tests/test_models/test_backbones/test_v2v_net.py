# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.models.backbones import V2VNet


def test_v2v_net():
    """Test V2VNet."""
    model = V2VNet(input_channels=17, output_channels=15)
    input = torch.randn(2, 17, 32, 32, 32)
    output = model(input)
    assert isinstance(output, tuple)
    assert output[-1].shape == (2, 15, 32, 32, 32)
