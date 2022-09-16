# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import PixelData

from mmpose.structures import MultilevelPixelData


class TestMultilevelPixelData(TestCase):

    def get_multi_level_pixel_data(self):
        metainfo = dict(num_keypoints=17)
        sizes = [(64, 48), (32, 24), (16, 12)]
        heatmaps = [np.random.rand(17, h, w) for h, w in sizes]
        masks = [torch.rand(1, h, w) for h, w in sizes]
        data = MultilevelPixelData(
            metainfo=metainfo, heatmaps=heatmaps, masks=masks)

        return data

    def test_init(self):

        data = self.get_multi_level_pixel_data()
        self.assertIn('num_keypoints', data)
        self.assertTrue(data.nlevel == 3)
        self.assertTrue(data.shape == ((64, 48), (32, 24), (16, 12)))
        self.assertTrue(isinstance(data[0], PixelData))

    def test_setter(self):
        # test `set_field`
        data = self.get_multi_level_pixel_data()
        sizes = [(64, 48), (32, 24), (16, 8)]
        offset_maps = [torch.rand(2, h, w) for h, w in sizes]
        data.offset_maps = offset_maps

        # test `to_tensor`
        data = self.get_multi_level_pixel_data()
        self.assertTrue(isinstance(data[0].heatmaps, np.ndarray))
        data = data.to_tensor()
        self.assertTrue(isinstance(data[0].heatmaps, torch.Tensor))

        # test `cpu`
        data = self.get_multi_level_pixel_data()
        self.assertTrue(isinstance(data[0].heatmaps, np.ndarray))
        self.assertTrue(isinstance(data[0].masks, torch.Tensor))
        self.assertTrue(data[0].masks.device.type == 'cpu')
        data = data.cpu()
        self.assertTrue(isinstance(data[0].heatmaps, np.ndarray))
        self.assertTrue(data[0].masks.device.type == 'cpu')

        # test `to`
        data = self.get_multi_level_pixel_data()
        self.assertTrue(data[0].masks.device.type == 'cpu')
        data = data.to('cpu')
        self.assertTrue(data[0].masks.device.type == 'cpu')

        # test `numpy`
        data = self.get_multi_level_pixel_data()
        self.assertTrue(isinstance(data[0].masks, torch.Tensor))
        data = data.numpy()
        self.assertTrue(isinstance(data[0].masks, np.ndarray))

    def test_deleter(self):

        data = self.get_multi_level_pixel_data()

        for key in ['heatmaps', 'masks']:
            self.assertIn(key, data)
            exec(f'del data.{key}')
            self.assertNotIn(key, data)
