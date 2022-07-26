# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.backbones import V2VNet


class TestV2Vnet(TestCase):

    def test_v2v_net(self):
        """Test V2VNet."""
        model = V2VNet(input_channels=17, output_channels=15)
        input = torch.randn(2, 17, 32, 32, 32)
        output = model(input)
        self.assertIsInstance(output, tuple)
        self.assertEqual(output[-1].shape, (2, 15, 32, 32, 32))
