# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmpose.utils import register_all_modules


class TestRegressionHead(TestCase):

    def setUp(self) -> None:
        register_all_modules()

        self.config = [
            ('regression_head wo-sigma',
             dict(
                 type='RegressionHead',
                 in_channels=1024,
                 num_joints=17,
                 output_sigma=False,
                 decoder=dict('RegressionLabel', input_size=(192, 256)))),
            ('regression_head with-sigma',
             dict(
                 type='RegressionHead',
                 in_channels=1024,
                 num_joints=17,
                 output_sigma=True,
                 decoder=dict('RegressionLabel', input_size=(192, 256))))
        ]
