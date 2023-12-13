# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.utils.transformer import GAUEncoder, SinePositionalEncoding


class TestSinePositionalEncoding(TestCase):

    def test_init(self):

        spe = SinePositionalEncoding(out_channels=128)
        self.assertTrue(hasattr(spe, 'dim_t'))
        self.assertFalse(spe.dim_t.requires_grad)
        self.assertEqual(spe.dim_t.size(0), 128 // 2)

        spe = SinePositionalEncoding(out_channels=128, learnable=True)
        self.assertTrue(spe.dim_t.requires_grad)

        spe = SinePositionalEncoding(out_channels=128, eval_size=10)
        self.assertTrue(hasattr(spe, 'pos_enc_10'))
        self.assertEqual(spe.pos_enc_10.size(-1), 128)

        spe = SinePositionalEncoding(
            out_channels=128, eval_size=(2, 3), spatial_dim=2)
        self.assertTrue(hasattr(spe, 'pos_enc_(2, 3)'))
        self.assertSequenceEqual(
            getattr(spe, 'pos_enc_(2, 3)').shape[-2:], (128, 2))

    def test_generate_speoding(self):

        # spatial_dim = 1
        spe = SinePositionalEncoding(out_channels=128)
        pos_enc = spe.generate_pos_encoding(size=10)
        self.assertSequenceEqual(pos_enc.shape, (10, 128))

        position = torch.arange(8)
        pos_enc = spe.generate_pos_encoding(position=position)
        self.assertSequenceEqual(pos_enc.shape, (8, 128))

        with self.assertRaises(AssertionError):
            pos_enc = spe.generate_pos_encoding(size=10, position=position)

        # spatial_dim = 2
        spe = SinePositionalEncoding(out_channels=128, spatial_dim=2)
        pos_enc = spe.generate_pos_encoding(size=10)
        self.assertSequenceEqual(pos_enc.shape, (100, 128, 2))

        pos_enc = spe.generate_pos_encoding(size=(5, 6))
        self.assertSequenceEqual(pos_enc.shape, (30, 128, 2))

        position = torch.arange(8).unsqueeze(1).repeat(1, 2)
        pos_enc = spe.generate_pos_encoding(position=position)
        self.assertSequenceEqual(pos_enc.shape, (8, 128, 2))

        with self.assertRaises(AssertionError):
            pos_enc = spe.generate_pos_encoding(size=10, position=position)

        with self.assertRaises(ValueError):
            pos_enc = spe.generate_pos_encoding(size=position)

    def test_apply_additional_pos_enc(self):

        # spatial_dim = 1
        spe = SinePositionalEncoding(out_channels=128)
        pos_enc = spe.generate_pos_encoding(size=10)
        feature = torch.randn(2, 3, 10, 128)
        out_feature = spe.apply_additional_pos_enc(feature, pos_enc,
                                                   spe.spatial_dim)
        self.assertSequenceEqual(feature.shape, out_feature.shape)

        # spatial_dim = 2
        spe = SinePositionalEncoding(out_channels=128 // 2, spatial_dim=2)
        pos_enc = spe.generate_pos_encoding(size=(2, 5))
        feature = torch.randn(2, 3, 10, 128)
        out_feature = spe.apply_additional_pos_enc(feature, pos_enc,
                                                   spe.spatial_dim)
        self.assertSequenceEqual(feature.shape, out_feature.shape)

    def test_apply_rotary_pos_enc(self):

        # spatial_dim = 1
        spe = SinePositionalEncoding(out_channels=128)
        pos_enc = spe.generate_pos_encoding(size=10)
        feature = torch.randn(2, 3, 10, 128)
        out_feature = spe.apply_rotary_pos_enc(feature, pos_enc,
                                               spe.spatial_dim)
        self.assertSequenceEqual(feature.shape, out_feature.shape)

        # spatial_dim = 2
        spe = SinePositionalEncoding(out_channels=128, spatial_dim=2)
        pos_enc = spe.generate_pos_encoding(size=(2, 5))
        feature = torch.randn(2, 3, 10, 128)
        out_feature = spe.apply_rotary_pos_enc(feature, pos_enc,
                                               spe.spatial_dim)
        self.assertSequenceEqual(feature.shape, out_feature.shape)


class TestGAUEncoder(TestCase):

    def test_init(self):
        gau = GAUEncoder(in_token_dims=64, out_token_dims=64)
        self.assertTrue(gau.shortcut)

        gau = GAUEncoder(in_token_dims=64, out_token_dims=64, dropout_rate=0.5)
        self.assertTrue(hasattr(gau, 'dropout'))

    def test_forward(self):
        gau = GAUEncoder(in_token_dims=64, out_token_dims=64)

        # compatibility with various dimension input
        feat = torch.randn(2, 3, 64)
        with torch.no_grad():
            out_feat = gau.forward(feat)
        self.assertSequenceEqual(feat.shape, out_feat.shape)

        feat = torch.randn(1, 2, 3, 64)
        with torch.no_grad():
            out_feat = gau.forward(feat)
        self.assertSequenceEqual(feat.shape, out_feat.shape)

        feat = torch.randn(1, 2, 3, 4, 64)
        with torch.no_grad():
            out_feat = gau.forward(feat)
        self.assertSequenceEqual(feat.shape, out_feat.shape)

        # positional encoding
        gau = GAUEncoder(
            s=32, in_token_dims=64, out_token_dims=64, pos_enc=True)
        feat = torch.randn(2, 3, 64)
        spe = SinePositionalEncoding(out_channels=32)
        pos_enc = spe.generate_pos_encoding(size=3)
        with torch.no_grad():
            out_feat = gau.forward(feat, pos_enc=pos_enc)
        self.assertSequenceEqual(feat.shape, out_feat.shape)

        gau = GAUEncoder(
            s=32,
            in_token_dims=64,
            out_token_dims=64,
            pos_enc=True,
            spatial_dim=2)
        feat = torch.randn(1, 2, 6, 64)
        spe = SinePositionalEncoding(out_channels=32, spatial_dim=2)
        pos_enc = spe.generate_pos_encoding(size=(2, 3))
        with torch.no_grad():
            out_feat = gau.forward(feat, pos_enc=pos_enc)
        self.assertSequenceEqual(feat.shape, out_feat.shape)

        # mask
        gau = GAUEncoder(in_token_dims=64, out_token_dims=64)

        # compatibility with various dimension input
        feat = torch.randn(2, 3, 64)
        mask = torch.rand(2, 3, 3)
        with torch.no_grad():
            out_feat = gau.forward(feat, mask=mask)
        self.assertSequenceEqual(feat.shape, out_feat.shape)
