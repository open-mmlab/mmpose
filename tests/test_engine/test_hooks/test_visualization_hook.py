# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.data import InstanceData

from mmpose.engine.hooks import PoseVisualizationHook
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


def _rand_poses(num_boxes, h, w):
    center = torch.rand(num_boxes, 2)
    offset = torch.rand(num_boxes, 5, 2) / 2.0

    pose = center[:, None, :] + offset.clip(0, 1)
    pose[:, :, 0] *= w
    pose[:, :, 1] *= h

    return pose


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        PoseLocalVisualizer.get_instance('visualizer')

        data_sample = PoseDataSample()
        data_sample.set_metainfo({
            'img_path':
            osp.join(
                osp.dirname(__file__), '../../data/coco/000000000785.jpg')
        })
        self.data_batch = [{'data_sample': data_sample}] * 2

        pred_instances = InstanceData()
        pred_instances.keypoints = _rand_poses(5, 10, 12)
        pred_instances.score = torch.rand((5, 5))
        pred_det_data_sample = PoseDataSample()
        pred_det_data_sample.pred_instances = pred_instances
        self.outputs = [pred_det_data_sample] * 2

    def test_after_val_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = PoseVisualizationHook()
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = PoseVisualizationHook(draw=True)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertEqual(hook._test_index, 2)

        # test
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        test_out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = PoseVisualizationHook(draw=False, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{test_out_dir}'))

        hook = PoseVisualizationHook(draw=True, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{test_out_dir}'))
        shutil.rmtree(f'{timestamp}')
