# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmpose.models import MultiModalSSAHead


def test_multi_modal_ssa_head():

    # substantialize head
    train_cfg = dict(ssa_start_epoch=10)
    head = MultiModalSSAHead(
        num_classes=25, modality=('rgb', 'depth'), train_cfg=train_cfg)

    head.set_train_epoch(11)
    assert head._train_epoch == 11
    assert head._train_epoch > head.start_epoch

    # forward
    img_metas = dict(modality=['rgb', 'depth'])
    feats = [torch.randn(2, 1024, 7, 7, 7) for _ in img_metas['modality']]
    labels = torch.randint(25, (2, ))

    logits = head(feats, img_metas)
    assert logits[0].shape == (2, 25, 7)

    losses = head.get_loss(logits, labels, feats)
    assert 'ce_loss' in losses
    assert 'ssa_loss' in losses
    assert (losses['ssa_loss'] == losses['ssa_loss']).all()  # check nan

    logits[0][0, 1], logits[1][0, 1], labels[0] = 1e5, 1e5, 1
    logits[0][1, 4], logits[1][1, 8], labels[1] = 1e5, 1e5, 8
    accuracy = head.get_accuracy(logits, labels, img_metas)
    assert 'acc_rgb' in accuracy
    assert 'acc_depth' in accuracy
    np.testing.assert_almost_equal(accuracy['acc_rgb'], 0.5)
    np.testing.assert_almost_equal(accuracy['acc_depth'], 1.0)
