# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.models.backbones import I3D


def test_i3d_backbone():
    """Test I3D backbone."""
    model = I3D()
    model.train()

    vids = torch.randn(1, 3, 16, 112, 112)
    feat = model(vids)
    assert feat.shape == (1, 1024, 2, 3, 3)

    model = I3D(expansion=0.5)
    model.train()

    vids = torch.randn(1, 3, 32, 224, 224)
    feat = model(vids)
    assert feat.shape == (1, 512, 4, 7, 7)
