# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones import DSTFormer
from mmpose.models.backbones.dstformer import AttentionBlock


class TestDSTFormer(TestCase):

    def test_attention_block(self):
        # BasicTemporalBlock with causal == False
        block = AttentionBlock(dim=256, num_heads=2)
        x = torch.rand(2, 17, 256)
        x_out = block(x)
        self.assertEqual(x_out.shape, torch.Size([2, 17, 256]))

    def test_DSTFormer(self):
        # Test DSTFormer with depth=2
        model = DSTFormer(in_channels=3, depth=2, seq_len=2)
        pose3d = torch.rand((1, 2, 17, 3))
        feat = model(pose3d)
        self.assertEqual(feat[0].shape, (2, 17, 256))

        # Test DSTFormer with depth=4 and qkv_bias=False
        model = DSTFormer(in_channels=3, depth=4, seq_len=2, qkv_bias=False)
        pose3d = torch.rand((1, 2, 17, 3))
        feat = model(pose3d)
        self.assertEqual(feat[0].shape, (2, 17, 256))

        # Test DSTFormer with depth=4 and att_fuse=False
        model = DSTFormer(in_channels=3, depth=4, seq_len=2, att_fuse=False)
        pose3d = torch.rand((1, 2, 17, 3))
        feat = model(pose3d)
        self.assertEqual(feat[0].shape, (2, 17, 256))
