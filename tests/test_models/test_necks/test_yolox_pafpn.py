# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.necks import YOLOXPAFPN


class TestYOLOXPAFPN(TestCase):

    def test_forward(self):
        in_channels = [128, 256, 512]
        out_channels = 256
        num_csp_blocks = 3

        model = YOLOXPAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_csp_blocks=num_csp_blocks)
        model.train()

        inputs = [
            torch.randn(1, c, 64 // (2**i), 64 // (2**i))
            for i, c in enumerate(in_channels)
        ]
        outputs = model(inputs)

        self.assertEqual(len(outputs), len(in_channels))
        for out in outputs:
            self.assertEqual(out.shape[1], out_channels)
