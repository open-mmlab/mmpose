# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmpose.models.heads import HeatmapHead


class TestHeatmapHead(TestCase):

    def test_init(self):
        # w/o deconv
        _ = HeatmapHead(
            in_channels=32, out_channels=17, deconv_out_channels=None)

        # w/ deconv and w/o conv
        _ = HeatmapHead(
            in_channels=32,
            out_channels=17,
            deconv_out_channels=(32, 32),
            deconv_kernel_sizes=(4, 4))

        # w/ both deconv and conv
        _ = HeatmapHead(
            in_channels=32,
            out_channels=17,
            deconv_out_channels=(32, 32),
            deconv_kernel_sizes=(4, 4),
            conv_out_channels=(32, ),
            conv_kernel_sizes=(1, ))
