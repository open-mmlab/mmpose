# Copyright (c) OpenMMLab. All rights reserved.

import torch
from addict import Dict

from mmpose.models.detectors import GestureRecognizer


def test_gesture_recognizer_forward():
    model_cfg = dict(
        type='GestureRecognizer',
        pretrained=None,
        modality=['rgb', 'depth'],
        backbone=dict(
            rgb=dict(
                type='I3D',
                in_channels=3,
                expansion=0.25,
            ),
            depth=dict(
                type='I3D',
                in_channels=1,
                expansion=0.25,
            ),
        ),
        cls_head=dict(
            type='MultiModalSSAHead',
            num_classes=25,
            avg_pool_kernel=(1, 2, 2),
            in_channels=256),
        train_cfg=dict(
            beta=2,
            lambda_=1e-3,
            ssa_start_epoch=10,
        ),
        test_cfg=dict(),
    )

    detector = GestureRecognizer(model_cfg['backbone'], None,
                                 model_cfg['cls_head'], model_cfg['train_cfg'],
                                 model_cfg['test_cfg'], model_cfg['modality'],
                                 model_cfg['pretrained'])
    detector.set_train_epoch(11)

    video = [torch.randn(1, 3, 16, 112, 112), torch.randn(1, 1, 16, 112, 112)]
    labels = torch.tensor([1]).long()
    img_metas = Dict()
    img_metas.data = dict(modality=['rgb', 'depth'])

    # Test forward train
    losses = detector.forward(video, labels, img_metas, return_loss=True)
    assert isinstance(losses, dict)
    assert 'ssa_loss' in losses

    # Test forward test
    with torch.no_grad():
        _ = detector.forward(
            video, labels, img_metas=img_metas, return_loss=False)
