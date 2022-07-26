# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones.hrformer import (HRFomerModule, HRFormer,
                                              HRFormerBlock)


class TestHrformer(TestCase):

    def test_hrformer_module(self):
        norm_cfg = dict(type='BN')
        block = HRFormerBlock
        # Test multiscale forward
        num_channles = (32, 64)
        num_inchannels = [c * block.expansion for c in num_channles]
        hrmodule = HRFomerModule(
            num_branches=2,
            block=block,
            num_blocks=(2, 2),
            num_inchannels=num_inchannels,
            num_channels=num_channles,
            num_heads=(1, 2),
            num_window_sizes=(7, 7),
            num_mlp_ratios=(4, 4),
            drop_paths=(0., 0.),
            norm_cfg=norm_cfg)

        feats = [
            torch.randn(1, num_inchannels[0], 64, 64),
            torch.randn(1, num_inchannels[1], 32, 32)
        ]
        feats = hrmodule(feats)

        self.assertGreater(len(str(hrmodule)), 0)
        self.assertEqual(len(feats), 2)
        self.assertEqual(feats[0].shape,
                         torch.Size([1, num_inchannels[0], 64, 64]))
        self.assertEqual(feats[1].shape,
                         torch.Size([1, num_inchannels[1], 32, 32]))

        # Test single scale forward
        num_channles = (32, 64)
        in_channels = [c * block.expansion for c in num_channles]
        hrmodule = HRFomerModule(
            num_branches=2,
            block=block,
            num_blocks=(2, 2),
            num_inchannels=num_inchannels,
            num_channels=num_channles,
            num_heads=(1, 2),
            num_window_sizes=(7, 7),
            num_mlp_ratios=(4, 4),
            drop_paths=(0., 0.),
            norm_cfg=norm_cfg,
            multiscale_output=False,
        )

        feats = [
            torch.randn(1, in_channels[0], 64, 64),
            torch.randn(1, in_channels[1], 32, 32)
        ]
        feats = hrmodule(feats)

        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].shape,
                         torch.Size([1, in_channels[0], 64, 64]))

        # Test single branch HRFormer module
        hrmodule = HRFomerModule(
            num_branches=1,
            block=block,
            num_blocks=(1, ),
            num_inchannels=[num_inchannels[0]],
            num_channels=[num_channles[0]],
            num_heads=(1, ),
            num_window_sizes=(7, ),
            num_mlp_ratios=(4, ),
            drop_paths=(0.1, ),
            norm_cfg=norm_cfg,
        )

        feats = [
            torch.randn(1, in_channels[0], 64, 64),
        ]
        feats = hrmodule(feats)

        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0].shape,
                         torch.Size([1, in_channels[0], 64, 64]))

        # Value tests
        kwargs = dict(
            num_branches=2,
            block=block,
            num_blocks=(2, 2),
            num_inchannels=num_inchannels,
            num_channels=num_channles,
            num_heads=(1, 2),
            num_window_sizes=(7, 7),
            num_mlp_ratios=(4, 4),
            drop_paths=(0.1, 0.1),
            norm_cfg=norm_cfg,
        )

        with self.assertRaises(ValueError):
            # len(num_blocks) should equal num_branches
            kwargs['num_blocks'] = [2, 2, 2]
            HRFomerModule(**kwargs)
        kwargs['num_blocks'] = [2, 2]

        with self.assertRaises(ValueError):
            # len(num_blocks) should equal num_branches
            kwargs['num_channels'] = [2]
            HRFomerModule(**kwargs)
        kwargs['num_channels'] = [2, 2]

        with self.assertRaises(ValueError):
            # len(num_blocks) should equal num_branches
            kwargs['num_inchannels'] = [2]
            HRFomerModule(**kwargs)
        kwargs['num_inchannels'] = [2, 2]

    def test_hrformer_backbone(self):
        norm_cfg = dict(type='BN')
        # only have 3 stages
        extra = dict(
            drop_path_rate=0.2,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='HRFORMERBLOCK',
                window_sizes=(7, 7),
                num_heads=(1, 2),
                mlp_ratios=(4, 4),
                num_blocks=(2, 2),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='HRFORMERBLOCK',
                window_sizes=(7, 7, 7),
                num_heads=(1, 2, 4),
                mlp_ratios=(4, 4, 4),
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='HRFORMERBLOCK',
                window_sizes=(7, 7, 7, 7),
                num_heads=(1, 2, 4, 8),
                mlp_ratios=(4, 4, 4, 4),
                num_blocks=(2, 2, 2, 2),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True))

        with self.assertRaises(ValueError):
            # len(num_blocks) should equal num_branches
            extra['stage4']['num_branches'] = 3
            HRFormer(extra=extra)
        extra['stage4']['num_branches'] = 4

        # Test HRFormer-S
        model = HRFormer(extra=extra, norm_cfg=norm_cfg)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feats = model(imgs)
        self.assertEqual(len(feats), 4)
        self.assertEqual(feats[0].shape, torch.Size([1, 32, 16, 16]))
        self.assertEqual(feats[3].shape, torch.Size([1, 256, 2, 2]))

        # Test single scale output and model
        # without relative position bias
        extra['stage4']['multiscale_output'] = False
        extra['with_rpe'] = False
        model = HRFormer(extra=extra, norm_cfg=norm_cfg)
        model.init_weights()
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feats = model(imgs)
        self.assertIsInstance(feats, tuple)
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[-1].shape, torch.Size([1, 32, 16, 16]))
