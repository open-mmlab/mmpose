# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.models.backbones2 import AlexNet


def test_alexnet_backbone():
    """Test alexnet backbone."""
    model = AlexNet(-1)
    model.train()

    imgs = torch.randn(1, 3, 256, 192)
    feat = model(imgs)
    assert isinstance(feat, tuple)
    assert feat[-1].shape == (1, 256, 7, 5)

    model = AlexNet(1)
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert isinstance(feat, tuple)
    assert feat[-1].shape == (1, 1)
