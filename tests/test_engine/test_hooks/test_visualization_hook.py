# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from mmengine.structures import InstanceData

from mmpose.engine.hooks import PoseVisualizationHook
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


def _rand_poses(num_boxes, h, w):
    center = np.random.rand(num_boxes, 2)
    offset = np.random.rand(num_boxes, 5, 2) / 2.0

    pose = center[:, None, :] + offset.clip(0, 1)
    pose[:, :, 0] *= w
    pose[:, :, 1] *= h

    return pose


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        PoseLocalVisualizer.get_instance('test_visualization_hook')

        data_sample = PoseDataSample()
        data_sample.set_metainfo({
            'img_path':
            osp.join(
                osp.dirname(__file__), '../../data/coco/000000000785.jpg')
        })
        self.data_batch = {'data_samples': [data_sample] * 2}

        pred_instances = InstanceData()
        pred_instances.keypoints = _rand_poses(5, 10, 12)
        pred_instances.score = np.random.rand(5, 5)
        pred_det_data_sample = data_sample.clone()
        pred_det_data_sample.pred_instances = pred_instances
        self.outputs = [pred_det_data_sample] * 2

    def test_after_val_iter(self):
        runner = MagicMock()
        runner.iter = 1
        runner.val_evaluator.dataset_meta = dict()
        hook = PoseVisualizationHook(interval=1, enable=True)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = MagicMock()
        runner.iter = 1
        hook = PoseVisualizationHook(enable=True)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertEqual(hook._test_index, 2)

        # test
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = PoseVisualizationHook(enable=False, out_dir=out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{out_dir}'))

        hook = PoseVisualizationHook(enable=True, out_dir=out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{out_dir}'))
        shutil.rmtree(f'{timestamp}')
