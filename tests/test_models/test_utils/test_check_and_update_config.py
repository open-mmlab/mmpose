# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmpose.models.utils import check_and_update_config


class TestCheckAndUpdateConfig(unittest.TestCase):

    def test_case_1(self):
        neck = None
        head = dict(
            type='HeatmapHead',
            in_channels=768,
            out_channels=17,
            deconv_out_channels=[],
            deconv_kernel_sizes=[],
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder='codec',
            align_corners=False,
            extra=dict(upsample=4, final_conv_kernel=3))
        neck, head = check_and_update_config(neck, head)

        self.assertDictEqual(
            neck,
            dict(
                type='FeatureMapProcessor',
                scale_factor=4.0,
                apply_relu=True,
            ))
        self.assertIn('final_layer', head)
        self.assertDictEqual(head['final_layer'],
                             dict(kernel_size=3, padding=1))
        self.assertNotIn('extra', head)
        self.assertNotIn('input_transform', head)
        self.assertNotIn('input_index', head)
        self.assertNotIn('align_corners', head)

    def test_case_2(self):
        neck = None
        head = dict(
            type='CIDHead',
            in_channels=(48, 96, 192, 384),
            num_keypoints=17,
            gfd_channels=48,
            input_transform='resize_concat',
            input_index=(0, 1, 2, 3),
            coupled_heatmap_loss=dict(
                type='FocalHeatmapLoss', loss_weight=1.0),
            decoupled_heatmap_loss=dict(
                type='FocalHeatmapLoss', loss_weight=4.0),
        )
        neck, head = check_and_update_config(neck, head)

        self.assertDictEqual(
            neck,
            dict(
                type='FeatureMapProcessor',
                concat=True,
                select_index=(0, 1, 2, 3),
            ))
        self.assertEqual(head['in_channels'], 720)
        self.assertNotIn('input_transform', head)
        self.assertNotIn('input_index', head)
        self.assertNotIn('align_corners', head)

    def test_case_3(self):
        neck = None
        head = dict(
            type='HeatmapHead',
            in_channels=(64, 128, 320, 512),
            out_channels=17,
            input_index=3,
            has_final_layer=False,
            loss=dict(type='KeypointMSELoss', use_target_weight=True))
        neck, head = check_and_update_config(neck, head)

        self.assertDictEqual(
            neck, dict(
                type='FeatureMapProcessor',
                select_index=3,
            ))
        self.assertEqual(head['in_channels'], 512)
        self.assertIn('final_layer', head)
        self.assertIsNone(head['final_layer'])
        self.assertNotIn('input_transform', head)
        self.assertNotIn('input_index', head)
        self.assertNotIn('align_corners', head)

    def test_case_4(self):
        neck = None
        head = dict(
            type='RTMCCHead',
            in_channels=768,
            out_channels=17,
            input_size='input_size',
            in_featuremap_size=(9, 12),
            simcc_split_ratio='simcc_split_ratio',
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False))
        neck, head_new = check_and_update_config(neck, head)

        self.assertIsNone(neck)
        self.assertDictEqual(head, head_new)


if __name__ == '__main__':
    unittest.main()
