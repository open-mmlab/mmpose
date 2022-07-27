# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from collections import defaultdict
from unittest import TestCase

import numpy as np
import torch
from mmengine.fileio import dump, load

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.metrics import PoseTrack18Metric


class TestPoseTrack18Metric(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ann_file = 'tests/data/posetrack18/annotations/'\
            'test_posetrack18_val.json'
        posetrack18_meta_info = dict(
            from_file='configs/_base_/datasets/posetrack18.py')
        self.posetrack18_dataset_meta = parse_pose_metainfo(
            posetrack18_meta_info)

        self.db = load(self.ann_file)

        self.topdown_data = self._convert_ann_to_topdown_batch_data()
        assert len(self.topdown_data) == 14
        self.bottomup_data = self._convert_ann_to_bottomup_batch_data()
        assert len(self.bottomup_data) == 3
        self.target = {
            'posetrack18/Head AP': 100.0,
            'posetrack18/Shou AP': 100.0,
            'posetrack18/Elb AP': 100.0,
            'posetrack18/Wri AP': 100.0,
            'posetrack18/Hip AP': 100.0,
            'posetrack18/Knee AP': 100.0,
            'posetrack18/Ankl AP': 100.0,
            'posetrack18/Total AP': 100.0,
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
        # test `format_only` option
        with self.assertRaisesRegex(
                AssertionError,
                '`outfile_prefix` can not be None when `format_only` is True'):
            PoseTrack18Metric(
                ann_file=self.ann_file, format_only=True, outfile_prefix=None)

    def test_topdown_evaluate(self):
        """test topdown-style posetrack18 metric evaluation."""
        posetrack18_metric = PoseTrack18Metric(
            ann_file=self.ann_file, outfile_prefix=f'{self.tmp_dir.name}/test')
        posetrack18_metric.dataset_meta = self.posetrack18_dataset_meta

        # process samples
        for batch, preds in self.topdown_data:
            posetrack18_metric.process(batch, preds)

        eval_results = posetrack18_metric.evaluate(size=len(self.topdown_data))

        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, '003418_mpii_test.json')))

    def test_bottomup_evaluate(self):
        """test bottomup-style posetrack18 metric evaluation."""
        posetrack18_metric = PoseTrack18Metric(
            ann_file=self.ann_file, outfile_prefix=f'{self.tmp_dir.name}/test')
        posetrack18_metric.dataset_meta = self.posetrack18_dataset_meta

        # process samples
        for batch, preds in self.bottomup_data:
            posetrack18_metric.process(batch, preds)

        eval_results = posetrack18_metric.evaluate(
            size=len(self.bottomup_data))
        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, '009473_mpii_test.json')))

    def test_other_methods(self):
        """test other useful methods."""
        # test `_sort_and_unique_bboxes` method
        posetrack18_metric = PoseTrack18Metric(ann_file=self.ann_file)
        posetrack18_metric.dataset_meta = self.posetrack18_dataset_meta
        # process samples
        for batch, preds in self.topdown_data:
            posetrack18_metric.process(batch, preds)
        # process one extra sample
        batch, preds = self.topdown_data[0]
        posetrack18_metric.process(batch, preds)
        # an extra sample
        eval_results = posetrack18_metric.evaluate(
            size=len(self.topdown_data) + 1)
        self.assertDictEqual(eval_results, self.target)

    def test_format_only(self):
        """test `format_only` option."""
        posetrack18_metric = PoseTrack18Metric(
            ann_file=self.ann_file,
            format_only=True,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        posetrack18_metric.dataset_meta = self.posetrack18_dataset_meta
        # process samples
        for batch, preds in self.topdown_data:
            posetrack18_metric.process(batch, preds)
        eval_results = posetrack18_metric.evaluate(size=len(self.topdown_data))
        self.assertDictEqual(eval_results, {})
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, '012834_mpii_test.json')))

        # test when gt annotations are absent
        db_ = load(self.ann_file)
        del db_['annotations']
        tmp_ann_file = osp.join(self.tmp_dir.name, 'temp_ann.json')
        dump(db_, tmp_ann_file, sort_keys=True, indent=4)
        with self.assertRaisesRegex(
                AssertionError,
                'Ground truth annotations are required for evaluation'):
            _ = PoseTrack18Metric(ann_file=tmp_ann_file, format_only=False)
