# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.models import PoseFormer
from mmpose.models.heads import PoseFormerHead


def test_poseformer():
    model = PoseFormer()
    model.train()

    x = torch.randn(1, 9, 17, 2)
    feat = model(x)

    assert feat.shape == (1, 1, 544)

    head = PoseFormerHead(loss_keypoint=dict(type='MPJPELoss'))
    head.train()

    out = head(feat)

    assert out.shape == (1, 17, 3)
