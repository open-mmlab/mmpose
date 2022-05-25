# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.data import BaseDataElement as PixelData
from mmengine.data import InstanceData

from mmpose.core import PoseDataSample


class TestPoseDataSample(TestCase):

    def get_pose_data_sample(self):
        # meta
        pose_meta = dict(
            img_shape=(600, 900, 3),  # [h, w, c]
            crop_size=(256, 192),  # [h, w]
            heatmap_size=(64, 48),  # [h, w]
        )
        # gt_instances
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.rand(1, 4)
        gt_instances.keypoints = torch.rand(1, 17, 2)
        gt_instances.keypoints_visible = torch.rand(1, 17, 1)

        # pred_instances
        pred_instances = InstanceData()
        pred_instances.keypoints = torch.rand(1, 17, 2)
        pred_instances.scores = torch.rand(1, 17, 1)

        # gt_fields
        gt_fields = PixelData()
        gt_fields.heatmaps = torch.rand(17, 64, 48)

        # pred_fields
        pred_fields = PixelData()
        pred_fields.heatmaps = torch.rand(17, 64, 48)

        data_sample = PoseDataSample(
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            gt_fields=gt_fields,
            pred_fields=pred_fields,
            metainfo=pose_meta)

        return data_sample

    @staticmethod
    def _equal(x, y):
        if type(x) != type(y):
            return False
        if isinstance(x, torch.Tensor):
            return torch.allclose(x, y)
        elif isinstance(x, np.ndarray):
            return np.allclose(x, y)
        else:
            return x == y

    def test_init(self):

        data_sample = self.get_pose_data_sample()
        self.assertIn('img_shape', data_sample)
        self.assertTrue(len(data_sample.gt_instances) == 1)

    def test_setter(self):

        data_sample = self.get_pose_data_sample()

        # test gt_instances
        data_sample.gt_instances = InstanceData()

        # test gt_fields
        data_sample.gt_fields = PixelData()

        # test pred_instances as pytorch tensor
        pred_instances_data = dict(
            keypoints=torch.rand(1, 17, 2), scores=torch.rand(1, 17, 1))
        data_sample.pred_instances = InstanceData(**pred_instances_data)

        self.assertTrue(
            self._equal(data_sample.pred_instances.keypoints,
                        pred_instances_data['keypoints']))
        self.assertTrue(
            self._equal(data_sample.pred_instances.scores,
                        pred_instances_data['scores']))

        # test pred_fields as numpy array
        pred_fields_data = dict(heatmaps=np.random.rand(17, 64, 48))
        data_sample.pred_fields = PixelData(**pred_fields_data)

        self.assertTrue(
            self._equal(data_sample.pred_fields.heatmaps,
                        pred_fields_data['heatmaps']))

        # test to_tensor
        data_sample = data_sample.to_tensor()
        self.assertTrue(
            self._equal(data_sample.pred_fields.heatmaps,
                        torch.from_numpy(pred_fields_data['heatmaps'])))

    def test_deleter(self):

        data_sample = self.get_pose_data_sample()

        for key in [
                'gt_instances', 'pred_instances', 'gt_fields', 'pred_fields'
        ]:
            self.assertIn(key, data_sample)
            exec(f'del data_sample.{key}')
            self.assertNotIn(key, data_sample)
