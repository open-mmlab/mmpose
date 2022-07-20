# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.models import builder


def test_v2v_net():
    """Test V2VNet."""
    cfg = dict(type='V2VNet', input_channels=17, output_channels=15),
    model = builder.build_backbone(*cfg)
    input = torch.randn(2, 17, 32, 32, 32)
    output = model(input)
    assert output.shape == (2, 15, 32, 32, 32)
