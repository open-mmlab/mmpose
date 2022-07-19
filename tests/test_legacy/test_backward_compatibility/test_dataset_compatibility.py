# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.datasets.datasets.base import (Kpt2dSviewRgbImgTopDownDataset,
                                           Kpt2dSviewRgbVidTopDownDataset,
                                           Kpt3dSviewRgbImgTopDownDataset)


class TestDatasetCompatibility(TestCase):

    def test_xywh2cs(self):

        base_dataset_list = [
            Kpt2dSviewRgbImgTopDownDataset, Kpt2dSviewRgbVidTopDownDataset,
            Kpt3dSviewRgbImgTopDownDataset
        ]
        for basetype in base_dataset_list:

            class DummyDataset(basetype):

                def __init__(self):
                    self.ann_info = dict(image_size=[192, 256])
                    self.test_mode = True

                def _get_db(self):
                    pass

                def evaluate(self, results, *args, **kwargs):
                    pass

            dataset = DummyDataset()

            with self.assertWarnsRegex(
                    DeprecationWarning,
                    'The ``_xywh2cs`` method will be deprecated'):
                bbox = np.array([0, 0, 100, 100], dtype=np.float32)
                _ = dataset._xywh2cs(*bbox)
