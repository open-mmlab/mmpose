# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch

from mmpose.models.heads.hybrid_heads.rtmo_head import RTMOHead


class TestRTMOHeadSwitchToDeploy(TestCase):
    """Tests for RTMOHead.switch_to_deploy.

    switch_to_deploy precomputes a prior anchor grid from dummy feature maps
    sized according to test_cfg['input_size'].  mmpose's input_size follows
    (W, H) order, while PyTorch tensor spatial dimensions are (H, W).
    Swapping the two is harmless for square inputs (W == H) but causes the
    prior grid to cover the wrong coordinate range for non-square models,
    shifting every detection to a wrong position at inference time.
    """

    def _get_head(self):
        return RTMOHead(
            num_keypoints=17,
            featmap_strides=(16, 32),
            head_module_cfg=dict(
                num_classes=1,
                in_channels=64,
                cls_feat_channels=64,
                channels_per_group=36,
                pose_vec_channels=64,
                widen_factor=1.0,
                stacked_convs=2,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='SiLU', inplace=True)),
            assigner=dict(
                type='SimOTAAssigner',
                dynamic_k_indicator='oks',
                oks_calculator=dict(
                    type='PoseOKS',
                    metainfo='configs/_base_/datasets/coco.py'),
                use_keypoints_for_center=True),
            prior_generator=dict(
                type='MlvlPointGenerator',
                centralize_points=True,
                strides=[16, 32]),
            dcc_cfg=dict(
                in_channels=64,
                feat_channels=32,
                num_bins=(64, 64),
                spe_channels=32,
                gau_cfg=dict(
                    s=64,
                    expansion_factor=2,
                    dropout_rate=0.0,
                    drop_path=0.0,
                    act_fn='SiLU',
                    pos_enc='add')),
            loss_cls=dict(
                type='VariFocalLoss',
                reduction='sum',
                use_target_weight=True,
                loss_weight=1.0),
            loss_bbox=dict(
                type='IoULoss',
                mode='square',
                eps=1e-16,
                reduction='sum',
                loss_weight=5.0),
            loss_oks=dict(
                type='OKSLoss',
                reduction='none',
                metainfo='configs/_base_/datasets/coco.py',
                loss_weight=30.0),
            loss_vis=dict(
                type='BCELoss',
                use_target_weight=True,
                reduction='mean',
                loss_weight=1.0))

    def _get_feats(self, input_size, batch_size=2, in_channels=64):
        """Return a list of dummy feature tensors matching ``input_size``."""
        W, H = input_size
        return [
            torch.rand(batch_size, in_channels, H // 16, W // 16),
            torch.rand(batch_size, in_channels, H // 32, W // 32),
        ]

    def test_switch_to_deploy_square_input_size(self):
        """Square input (W == H) should produce consistent x/y prior ranges."""
        W = H = 640
        head = self._get_head()
        head.switch_to_deploy(test_cfg=dict(input_size=(W, H)))

        x_max = head.flatten_priors[:, 0].max().item()
        y_max = head.flatten_priors[:, 1].max().item()

        # For a square input both axes cover the same range
        self.assertLessEqual(x_max, W)
        self.assertLessEqual(y_max, H)
        # Priors must actually reach near the image boundary
        self.assertGreater(x_max, 0)
        self.assertGreater(y_max, 0)

    def test_switch_to_deploy_non_square_input_size(self):
        """Non-square input (W != H) must keep x priors within [0, W] and
        y priors within [0, H].

        mmpose's input_size is (W, H), but torch.rand takes spatial dims as
        (H, W).  If the indices are passed without swapping, the feature maps
        get shape (W/stride, H/stride) instead of (H/stride, W/stride).
        MlvlPointGenerator then produces x priors up to H and y priors up to
        W — exactly backwards.  For input_size=(640, 384) this means x only
        reaches ~384 and y reaches ~640, causing all detections to appear
        in the wrong region of the image.
        """
        W, H = 640, 384
        head = self._get_head()
        head.switch_to_deploy(test_cfg=dict(input_size=(W, H)))

        # Priors are (x, y, stride) triples.
        # Column 0 is the x (width) axis; column 1 is the y (height) axis.
        x_max = head.flatten_priors[:, 0].max().item()
        y_max = head.flatten_priors[:, 1].max().item()

        self.assertLessEqual(
            x_max, W,
            f'x priors exceed input width {W}: got {x_max:.0f}')
        self.assertLessEqual(
            y_max, H,
            f'y priors exceed input height {H}: got {y_max:.0f}')

        # Key assertion: x priors must span close to W (640), not only H
        # (384).  If input_size[0] and input_size[1] are swapped when calling
        # torch.rand inside switch_to_deploy, x_max will be ~384 instead.
        self.assertGreater(
            x_max, H,
            f'x priors only reach {x_max:.0f}; expected close to W={W}. '
            f'Probable cause: input_size passed to torch.rand in (H, W) '
            f'order instead of the required (W, H) — swap the indices in '
            f'switch_to_deploy.')


if __name__ == '__main__':
    unittest.main()
