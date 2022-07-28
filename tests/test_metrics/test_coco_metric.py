# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
from collections import defaultdict
from unittest import TestCase

import numpy as np
import torch
from mmengine.fileio import dump, load

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.metrics import CocoMetric


class TestCocoMetric(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ann_file = 'tests/data/coco/test_coco.json'
        coco_meta_info = dict(from_file='configs/_base_/datasets/coco.py')
        self.coco_dataset_meta = parse_pose_metainfo(coco_meta_info)

        self.db = load(self.ann_file)

        self.topdown_data = self._convert_ann_to_topdown_batch_data()
        assert len(self.topdown_data) == 14
        self.bottomup_data = self._convert_ann_to_bottomup_batch_data()
        assert len(self.bottomup_data) == 4
        self.target = {
            'coco/AP': 1.0,
            'coco/AP .5': 1.0,
            'coco/AP .75': 1.0,
            'coco/AP (M)': 1.0,
            'coco/AP (L)': 1.0,
            'coco/AR': 1.0,
            'coco/AR .5': 1.0,
            'coco/AR .75': 1.0,
            'coco/AR (M)': 1.0,
            'coco/AR (L)': 1.0,
        }

    def _convert_ann_to_topdown_batch_data(self):
        """Convert annotations to topdown-style batch data."""
        data = []
        for ann in self.db['annotations']:
            w, h = ann['bbox'][2], ann['bbox'][3]
            data_batch = {}
            data_batch['inputs'] = None
            data_batch['data_sample'] = {
                'id': ann['id'],
                'img_id': ann['image_id'],
                'bbox_scales': np.array([w * 1.25, h * 1.25]).reshape(-1, 2),
            }
            predictions = {}
            predictions['pred_instances'] = {
                'keypoints': torch.tensor(ann['keypoints']).reshape(
                    (1, -1, 3)),
                'scores': torch.ones((1, 1))
            }
            # batch size = 1
            data_batch = [data_batch]
            predictions = [predictions]
            data.append((data_batch, predictions))
        return data

    def _convert_ann_to_bottomup_batch_data(self):
        """Convert annotations to bottomup-style batch data."""
        img2ann = defaultdict(list)
        for ann in self.db['annotations']:
            img2ann[ann['image_id']].append(ann)

        data = []
        for img_id, anns in img2ann.items():
            data_batch = {}
            data_batch['inputs'] = None
            data_batch['data_sample'] = {
                'id': [ann['id'] for ann in anns],
                'img_id': img_id,
            }
            predictions = {}
            predictions['pred_instances'] = {
                'keypoints':
                torch.tensor([ann['keypoints'] for ann in anns]).reshape(
                    (len(anns), -1, 3)),
                'scores':
                torch.ones((len(anns), 1))
            }
            # batch size = 1
            data_batch = [data_batch]
            predictions = [predictions]
            data.append((data_batch, predictions))
        return data

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        """test metric init method."""
        # test score_mode option
        with self.assertRaisesRegex(ValueError,
                                    '`score_mode` should be one of'):
            _ = CocoMetric(ann_file=self.ann_file, score_mode='keypoint')

        # test nms_mode option
        with self.assertRaisesRegex(ValueError, '`nms_mode` should be one of'):
            _ = CocoMetric(ann_file=self.ann_file, nms_mode='invalid')

        # test format_only option
        with self.assertRaisesRegex(
                AssertionError,
                '`outfile_prefix` can not be None when `format_only` is True'):
            _ = CocoMetric(
                ann_file=self.ann_file, format_only=True, outfile_prefix=None)

    def test_topdown_evaluate(self):
        """test topdown-style COCO metric evaluation."""
        # case 1: score_mode='bbox', nms_mode='none'
        coco_metric = CocoMetric(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox',
            nms_mode='none')
        coco_metric.dataset_meta = self.coco_dataset_meta

        # process samples
        for batch, preds in self.topdown_data:
            coco_metric.process(batch, preds)

        eval_results = coco_metric.evaluate(size=len(self.topdown_data))

        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

        # case 2: score_mode='bbox_keypoint', nms_mode='oks_nms'
        coco_metric = CocoMetric(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox_keypoint',
            nms_mode='oks_nms')
        coco_metric.dataset_meta = self.coco_dataset_meta

        # process samples
        for batch, preds in self.topdown_data:
            coco_metric.process(batch, preds)

        eval_results = coco_metric.evaluate(size=len(self.topdown_data))

        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

        # case 3: score_mode='bbox_rle', nms_mode='soft_oks_nms'
        coco_metric = CocoMetric(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox_rle',
            nms_mode='soft_oks_nms')
        coco_metric.dataset_meta = self.coco_dataset_meta

        # process samples
        for batch, preds in self.topdown_data:
            coco_metric.process(batch, preds)

        eval_results = coco_metric.evaluate(size=len(self.topdown_data))

        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

    def test_bottomup_evaluate(self):
        """test bottomup-style COCO metric evaluation."""
        # case1: score_mode='bbox', nms_mode='none'
        coco_metric = CocoMetric(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox',
            nms_mode='none')
        coco_metric.dataset_meta = self.coco_dataset_meta

        # process samples
        for batch, preds in self.bottomup_data:
            coco_metric.process(batch, preds)

        eval_results = coco_metric.evaluate(size=len(self.bottomup_data))
        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

    def test_other_methods(self):
        """test other useful methods."""
        # test `_sort_and_unique_bboxes` method
        coco_metric = CocoMetric(
            ann_file=self.ann_file, score_mode='bbox', nms_mode='none')
        coco_metric.dataset_meta = self.coco_dataset_meta
        # process samples
        for batch, preds in self.topdown_data:
            coco_metric.process(batch, preds)
        # process one extra sample
        batch, preds = self.topdown_data[0]
        coco_metric.process(batch, preds)
        # an extra sample
        eval_results = coco_metric.evaluate(size=len(self.topdown_data) + 1)
        self.assertDictEqual(eval_results, self.target)

    def test_format_only(self):
        """test `format_only` option."""
        coco_metric = CocoMetric(
            ann_file=self.ann_file,
            format_only=True,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox_keypoint',
            nms_mode='oks_nms')
        coco_metric.dataset_meta = self.coco_dataset_meta
        # process one sample
        batch, preds = self.topdown_data[0]
        coco_metric.process(batch, preds)
        eval_results = coco_metric.evaluate(size=1)
        self.assertDictEqual(eval_results, {})
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

        # test when gt annotations are absent
        db_ = load(self.ann_file)
        del db_['annotations']
        tmp_ann_file = osp.join(self.tmp_dir.name, 'temp_ann.json')
        dump(db_, tmp_ann_file, sort_keys=True, indent=4)
        with self.assertRaisesRegex(
                AssertionError,
                'Ground truth annotations are required for evaluation'):
            _ = CocoMetric(ann_file=tmp_ann_file, format_only=False)

    def test_topdown_alignment(self):
        """Test whether the output of CocoMetric and the original
        TopDownCocoDataset are the same."""
        data = []
        for ann in self.db['annotations']:
            w, h = ann['bbox'][2], ann['bbox'][3]
            data_batch = {}
            data_batch['inputs'] = None
            data_batch['data_sample'] = {
                'id': ann['id'],
                'img_id': ann['image_id'],
                'bbox_scales': np.array([w * 1.25, h * 1.25]).reshape(-1, 2),
            }
            predictions = {}
            keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(1, 17, 3)
            keypoints[0, :, 0] = keypoints[0, :, 0] * 0.98
            keypoints[0, :, 1] = keypoints[0, :, 1] * 1.02
            keypoints[0, :, 2] = keypoints[0, :, 2] * 0.8

            predictions['pred_instances'] = {
                'keypoints': torch.tensor(keypoints).reshape((1, -1, 3)),
                'scores': torch.ones((1, 1)) * 0.98
            }
            # batch size = 1
            data_batch = [data_batch]
            predictions = [predictions]
            data.append((data_batch, predictions))

        # case 1:
        # typical setting: score_mode='bbox_keypoint', nms_mode='oks_nms'
        coco_metric = CocoMetric(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox_keypoint',
            nms_mode='oks_nms')
        coco_metric.dataset_meta = self.coco_dataset_meta

        # process samples
        for batch, preds in data:
            coco_metric.process(batch, preds)

        eval_results = coco_metric.evaluate(size=len(data))

        target = {
            'coco/AP': 0.5287458745874587,
            'coco/AP .5': 0.9042904290429042,
            'coco/AP .75': 0.5009900990099009,
            'coco/AP (M)': 0.42475247524752474,
            'coco/AP (L)': 0.6219554455445544,
            'coco/AR': 0.5833333333333333,
            'coco/AR .5': 0.9166666666666666,
            'coco/AR .75': 0.5833333333333334,
            'coco/AR (M)': 0.44000000000000006,
            'coco/AR (L)': 0.6857142857142857,
        }

        for key in eval_results.keys():
            self.assertAlmostEqual(eval_results[key], target[key])

        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

        # case 2: score_mode='bbox_rle', nms_mode='oks_nms'
        coco_metric = CocoMetric(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox_rle',
            nms_mode='oks_nms')
        coco_metric.dataset_meta = self.coco_dataset_meta

        # process samples
        for batch, preds in data:
            coco_metric.process(batch, preds)

        eval_results = coco_metric.evaluate(size=len(data))

        target = {
            'coco/AP': 0.5004950495049505,
            'coco/AP .5': 0.8836633663366337,
            'coco/AP .75': 0.4679867986798679,
            'coco/AP (M)': 0.42475247524752474,
            'coco/AP (L)': 0.5814108910891089,
            'coco/AR': 0.5833333333333333,
            'coco/AR .5': 0.9166666666666666,
            'coco/AR .75': 0.5833333333333334,
            'coco/AR (M)': 0.44000000000000006,
            'coco/AR (L)': 0.6857142857142857,
        }

        for key in eval_results.keys():
            self.assertAlmostEqual(eval_results[key], target[key])

        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

        data = []
        anns = self.db['annotations']
        for i, ann in enumerate(anns):
            w, h = ann['bbox'][2], ann['bbox'][3]
            data_batch0 = {}
            data_batch0['inputs'] = None
            data_batch0['data_sample'] = {
                'id': ann['id'],
                'img_id': ann['image_id'],
                'bbox_scales': np.array([w * 1.25, h * 1.25]).reshape(-1, 2),
            }
            predictions0 = {}
            keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(1, 17, 3)
            keypoints[0, :, 0] = keypoints[0, :, 0] * (1 - i / 100)
            keypoints[0, :, 1] = keypoints[0, :, 1] * (1 + i / 100)
            keypoints[0, :, 2] = keypoints[0, :, 2] * (1 - i / 100)

            predictions0['pred_instances'] = {
                'keypoints': torch.tensor(keypoints).reshape((1, -1, 3)),
                'scores': torch.ones((1, 1))
            }

            data_batch1 = {}
            data_batch1['inputs'] = None
            data_batch1['data_sample'] = {
                'id': ann['id'] + 1,
                'img_id': ann['image_id'],
                'bbox_scales': np.array([w * 1.25, h * 1.25]).reshape(-1, 2),
            }
            predictions1 = {}
            keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(1, 17, 3)
            keypoints[0, :, 0] = keypoints[0, :, 0] * (1 + i / 100)
            keypoints[0, :, 1] = keypoints[0, :, 1] * (1 - i / 100)
            keypoints[0, :, 2] = keypoints[0, :, 2] * (1 - 2 * i / 100)

            predictions1['pred_instances'] = {
                'keypoints':
                torch.tensor(copy.deepcopy(keypoints)).reshape((1, -1, 3)),
                'scores':
                torch.ones((1, 1)) * (1 - 2 * i / 100)
            }

            # batch size = 2
            data_batch = [data_batch0, data_batch1]
            predictions = [predictions0, predictions1]
            data.append((data_batch, predictions))

        # case 3: score_mode='bbox_keypoint', nms_mode='oks_nms'
        coco_metric = CocoMetric(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox_keypoint',
            keypoint_score_thr=0.2,
            nms_thr=0.9,
            nms_mode='oks_nms')
        coco_metric.dataset_meta = self.coco_dataset_meta

        # process samples
        for batch, preds in data:
            coco_metric.process(batch, preds)

        eval_results = coco_metric.evaluate(size=len(data) * 2)

        target = {
            'coco/AP': 0.19344059405940592,
            'coco/AP .5': 0.297029702970297,
            'coco/AP .75': 0.10891089108910887,
            'coco/AP (M)': 0.0,
            'coco/AP (L)': 0.3329561527581329,
            'coco/AR': 0.2416666666666666,
            'coco/AR .5': 0.3333333333333333,
            'coco/AR .75': 0.16666666666666666,
            'coco/AR (M)': 0.0,
            'coco/AR (L)': 0.41428571428571426,
        }

        for key in eval_results.keys():
            self.assertAlmostEqual(eval_results[key], target[key])

        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))
