# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmpose.engine.hooks import BadCaseAnalysisHook
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


def _rand_poses(num_boxes, kpt_num, h, w):
    center = np.random.rand(num_boxes, 2)
    offset = np.random.rand(num_boxes, kpt_num, 2) / 2.0

    pose = center[:, None, :] + offset.clip(0, 1)
    pose[:, :, 0] *= w
    pose[:, :, 1] *= h

    return pose


class TestBadCaseHook(TestCase):

    def setUp(self) -> None:
        kpt_num = 16
        PoseLocalVisualizer.get_instance('test_badcase_hook')

        data_sample = PoseDataSample()
        data_sample.set_metainfo({
            'img_path':
            osp.join(
                osp.dirname(__file__), '../../data/coco/000000000785.jpg')
        })
        self.data_batch = {'data_samples': [data_sample] * 2}

        pred_det_data_sample = data_sample.clone()
        pred_instances = InstanceData()
        pred_instances.keypoints = _rand_poses(1, kpt_num, 10, 12)
        pred_det_data_sample.pred_instances = pred_instances

        gt_instances = InstanceData()
        gt_instances.keypoints = _rand_poses(1, kpt_num, 10, 12)
        gt_instances.keypoints_visible = np.ones((1, kpt_num))
        gt_instances.head_size = np.random.rand(1, 1)
        gt_instances.bboxes = np.random.rand(1, 4)
        pred_det_data_sample.gt_instances = gt_instances
        self.outputs = [pred_det_data_sample] * 2

    def test_after_test_iter(self):
        runner = MagicMock()
        runner.iter = 1

        # test
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = BadCaseAnalysisHook(enable=False, out_dir=out_dir)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{out_dir}'))

        hook = BadCaseAnalysisHook(
            enable=True,
            out_dir=out_dir,
            metric_type='loss',
            metric=ConfigDict(type='KeypointMSELoss'),
            badcase_thr=-1,  # is_badcase = True
        )
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertEqual(hook._test_index, 2)
        self.assertTrue(osp.exists(f'{timestamp}/1/{out_dir}'))
        # same image and preds/gts, so onlu one file
        self.assertTrue(len(os.listdir(f'{timestamp}/1/{out_dir}')) == 1)

        hook.after_test_epoch(runner)
        self.assertTrue(osp.exists(f'{timestamp}/1/{out_dir}/results.json'))
        shutil.rmtree(f'{timestamp}')

        hook = BadCaseAnalysisHook(
            enable=True,
            out_dir=out_dir,
            metric_type='accuracy',
            metric=ConfigDict(type='MpiiPCKAccuracy'),
            badcase_thr=-1,  # is_badcase = False
        )
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
        self.assertTrue(osp.exists(f'{timestamp}/1/{out_dir}'))
        self.assertTrue(len(os.listdir(f'{timestamp}/1/{out_dir}')) == 0)
        shutil.rmtree(f'{timestamp}')


if __name__ == '__main__':
    test = TestBadCaseHook()
    test.setUp()
    test.test_after_test_iter()
