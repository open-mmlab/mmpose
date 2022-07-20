# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmpose.utils import register_all_modules


class TestTopdownPoseEstimator(TestCase):

    def setUp(self) -> None:
        register_all_modules()
