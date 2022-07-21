# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones.swin import SwinBlock, SwinTransformer


class TestSwin(TestCase):

    def test_swin_block(self):
        # test SwinBlock structure and forward
        block = SwinBlock(embed_dims=64, num_heads=4, feedforward_channels=256)
        self.assertEqual(block.ffn.embed_dims, 64)
        self.assertEqual(block.attn.w_msa.num_heads, 4)
        self.assertEqual(block.ffn.feedforward_channels, 256)
        x = torch.randn(1, 56 * 56, 64)
        x_out = block(x, (56, 56))
        self.assertEqual(x_out.shape, torch.Size([1, 56 * 56, 64]))

        # Test BasicBlock with checkpoint forward
        block = SwinBlock(
            embed_dims=64, num_heads=4, feedforward_channels=256, with_cp=True)
        self.assertTrue(block.with_cp)
        x = torch.randn(1, 56 * 56, 64)
        x_out = block(x, (56, 56))
        self.assertEqual(x_out.shape, torch.Size([1, 56 * 56, 64]))

    def test_swin_transformer(self):
        """Test Swin Transformer backbone."""

        with self.assertRaises(AssertionError):
            # Because swin uses non-overlapping patch embed, so the stride of
            # patch embed must be equal to patch size.
            SwinTransformer(strides=(2, 2, 2, 2), patch_size=4)

        # test pretrained image size
        with self.assertRaises(AssertionError):
            SwinTransformer(pretrain_img_size=(224, 224, 224))

        # Test absolute position embedding
        temp = torch.randn((1, 3, 224, 224))
        model = SwinTransformer(pretrain_img_size=224, use_abs_pos_embed=True)
        model.init_weights()
        model(temp)

        # Test patch norm
        model = SwinTransformer(patch_norm=False)
        model(temp)

        # Test normal inference
        temp = torch.randn((1, 3, 32, 32))
        model = SwinTransformer()
        outs = model(temp)
        self.assertEqual(outs[0].shape, (1, 96, 8, 8))
        self.assertEqual(outs[1].shape, (1, 192, 4, 4))
        self.assertEqual(outs[2].shape, (1, 384, 2, 2))
        self.assertEqual(outs[3].shape, (1, 768, 1, 1))

        # Test abnormal inference size
        temp = torch.randn((1, 3, 31, 31))
        model = SwinTransformer()
        outs = model(temp)
        self.assertEqual(outs[0].shape, (1, 96, 8, 8))
        self.assertEqual(outs[1].shape, (1, 192, 4, 4))
        self.assertEqual(outs[2].shape, (1, 384, 2, 2))
        self.assertEqual(outs[3].shape, (1, 768, 1, 1))

        # Test abnormal inference size
        temp = torch.randn((1, 3, 112, 137))
        model = SwinTransformer()
        outs = model(temp)
        self.assertEqual(outs[0].shape, (1, 96, 28, 35))
        self.assertEqual(outs[1].shape, (1, 192, 14, 18))
        self.assertEqual(outs[2].shape, (1, 384, 7, 9))
        self.assertEqual(outs[3].shape, (1, 768, 4, 5))

        model = SwinTransformer(frozen_stages=4)
        model.train()
        for p in model.parameters():
            self.assertFalse(p.requires_grad)
