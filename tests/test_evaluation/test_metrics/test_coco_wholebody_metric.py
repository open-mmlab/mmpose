# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
from collections import defaultdict
from unittest import TestCase

import numpy as np
from mmengine.fileio import dump, load
from xtcocotools.coco import COCO

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.evaluation.metrics import CocoWholeBodyMetric


class TestCocoWholeBodyMetric(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.tmp_dir = tempfile.TemporaryDirectory()

        self.ann_file_coco = 'tests/data/coco/test_coco_wholebody.json'
        meta_info_coco = dict(
            from_file='configs/_base_/datasets/coco_wholebody.py')
        self.dataset_meta_coco = parse_pose_metainfo(meta_info_coco)
        self.coco = COCO(self.ann_file_coco)
        self.dataset_meta_coco['CLASSES'] = self.coco.loadCats(
            self.coco.getCatIds())

        self.topdown_data_coco = self._convert_ann_to_topdown_batch_data(
            self.ann_file_coco)
        assert len(self.topdown_data_coco) == 14
        self.bottomup_data_coco = self._convert_ann_to_bottomup_batch_data(
            self.ann_file_coco)
        assert len(self.bottomup_data_coco) == 4
        self.target_coco = {
            'coco-wholebody/AP': 1.0,
            'coco-wholebody/AP .5': 1.0,
            'coco-wholebody/AP .75': 1.0,
            'coco-wholebody/AP (M)': 1.0,
            'coco-wholebody/AP (L)': 1.0,
            'coco-wholebody/AR': 1.0,
            'coco-wholebody/AR .5': 1.0,
            'coco-wholebody/AR .75': 1.0,
            'coco-wholebody/AR (M)': 1.0,
            'coco-wholebody/AR (L)': 1.0,
        }

    def _convert_ann_to_topdown_batch_data(self, ann_file):
        """Convert annotations to topdown-style batch data."""
        topdown_data = []
        db = load(ann_file)
        imgid2info = dict()
        for img in db['images']:
            imgid2info[img['id']] = img
        for ann in db['annotations']:
            w, h = ann['bbox'][2], ann['bbox'][3]
            bboxes = np.array(ann['bbox'], dtype=np.float32).reshape(-1, 4)
            bbox_scales = np.array([w * 1.25, h * 1.25]).reshape(-1, 2)
            _keypoints = np.array(ann['keypoints'] + ann['foot_kpts'] +
                                  ann['face_kpts'] + ann['lefthand_kpts'] +
                                  ann['righthand_kpts']).reshape(1, -1, 3)

            gt_instances = {
                'bbox_scales': bbox_scales,
                'bbox_scores': np.ones((1, ), dtype=np.float32),
                'bboxes': bboxes,
            }
            pred_instances = {
                'keypoints': _keypoints[..., :2],
                'keypoint_scores': _keypoints[..., -1],
            }

            data = {'inputs': None}
            data_sample = {
                'id': ann['id'],
                'img_id': ann['image_id'],
                'category_id': ann.get('category_id', 1),
                'gt_instances': gt_instances,
                'pred_instances': pred_instances,
                # dummy image_shape for testing
                'ori_shape': [640, 480],
                # store the raw annotation info to test without ann_file
                'raw_ann_info': copy.deepcopy(ann),
            }

            # batch size = 1
            data_batch = [data]
            data_samples = [data_sample]
            topdown_data.append((data_batch, data_samples))

        return topdown_data

    def _convert_ann_to_bottomup_batch_data(self, ann_file):
        """Convert annotations to bottomup-style batch data."""
        img2ann = defaultdict(list)
        db = load(ann_file)
        for ann in db['annotations']:
            img2ann[ann['image_id']].append(ann)

        bottomup_data = []
        for img_id, anns in img2ann.items():
            _keypoints = []
            for ann in anns:
                _keypoints.append(ann['keypoints'] + ann['foot_kpts'] +
                                  ann['face_kpts'] + ann['lefthand_kpts'] +
                                  ann['righthand_kpts'])
            keypoints = np.array(_keypoints).reshape((len(anns), -1, 3))

            gt_instances = {
                'bbox_scores': np.ones((len(anns)), dtype=np.float32)
            }

            pred_instances = {
                'keypoints': keypoints[..., :2],
                'keypoint_scores': keypoints[..., -1],
            }

            data = {'inputs': None}
            data_sample = {
                'id': [ann['id'] for ann in anns],
                'img_id': img_id,
                'gt_instances': gt_instances,
                'pred_instances': pred_instances
            }

            # batch size = 1
            data_batch = [data]
            data_samples = [data_sample]
            bottomup_data.append((data_batch, data_samples))
        return bottomup_data

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        """test metric init method."""
        # test score_mode option
        with self.assertRaisesRegex(ValueError,
                                    '`score_mode` should be one of'):
            _ = CocoWholeBodyMetric(
                ann_file=self.ann_file_coco, score_mode='invalid')

        # test nms_mode option
        with self.assertRaisesRegex(ValueError, '`nms_mode` should be one of'):
            _ = CocoWholeBodyMetric(
                ann_file=self.ann_file_coco, nms_mode='invalid')

        # test format_only option
        with self.assertRaisesRegex(
                AssertionError,
                '`outfile_prefix` can not be None when `format_only` is True'):
            _ = CocoWholeBodyMetric(
                ann_file=self.ann_file_coco,
                format_only=True,
                outfile_prefix=None)

    def test_other_methods(self):
        """test other useful methods."""
        # test `_sort_and_unique_bboxes` method
        metric_coco = CocoWholeBodyMetric(
            ann_file=self.ann_file_coco, score_mode='bbox', nms_mode='none')
        metric_coco.dataset_meta = self.dataset_meta_coco
        # process samples
        for data_batch, data_samples in self.topdown_data_coco:
            metric_coco.process(data_batch, data_samples)
        # process one extra sample
        data_batch, data_samples = self.topdown_data_coco[0]
        metric_coco.process(data_batch, data_samples)
        # an extra sample
        eval_results = metric_coco.evaluate(
            size=len(self.topdown_data_coco) + 1)
        self.assertDictEqual(eval_results, self.target_coco)

    def test_format_only(self):
        """test `format_only` option."""
        metric_coco = CocoWholeBodyMetric(
            ann_file=self.ann_file_coco,
            format_only=True,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox_keypoint',
            nms_mode='oks_nms')
        metric_coco.dataset_meta = self.dataset_meta_coco
        # process one sample
        data_batch, data_samples = self.topdown_data_coco[0]
        metric_coco.process(data_batch, data_samples)
        eval_results = metric_coco.evaluate(size=1)
        self.assertDictEqual(eval_results, {})
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

        # test when gt annotations are absent
        db_ = load(self.ann_file_coco)
        del db_['annotations']
        tmp_ann_file = osp.join(self.tmp_dir.name, 'temp_ann.json')
        dump(db_, tmp_ann_file, sort_keys=True, indent=4)
        with self.assertRaisesRegex(
                AssertionError,
                'Ground truth annotations are required for evaluation'):
            _ = CocoWholeBodyMetric(ann_file=tmp_ann_file, format_only=False)

    def test_bottomup_evaluate(self):
        """test bottomup-style COCO metric evaluation."""
        # case1: score_mode='bbox', nms_mode='none'
        metric_coco = CocoWholeBodyMetric(
            ann_file=self.ann_file_coco,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox',
            nms_mode='none')
        metric_coco.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in self.bottomup_data_coco:
            metric_coco.process(data_batch, data_samples)

        eval_results = metric_coco.evaluate(size=len(self.bottomup_data_coco))
        self.assertDictEqual(eval_results, self.target_coco)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

    def test_topdown_evaluate(self):
        """test topdown-style COCO metric evaluation."""
        # case 1: score_mode='bbox', nms_mode='none'
        metric_coco = CocoWholeBodyMetric(
            ann_file=self.ann_file_coco,
            outfile_prefix=f'{self.tmp_dir.name}/test1',
            score_mode='bbox',
            nms_mode='none')
        metric_coco.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in self.topdown_data_coco:
            metric_coco.process(data_batch, data_samples)

        eval_results = metric_coco.evaluate(size=len(self.topdown_data_coco))

        self.assertDictEqual(eval_results, self.target_coco)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test1.keypoints.json')))

        # case 2: score_mode='bbox_keypoint', nms_mode='oks_nms'
        metric_coco = CocoWholeBodyMetric(
            ann_file=self.ann_file_coco,
            outfile_prefix=f'{self.tmp_dir.name}/test2',
            score_mode='bbox_keypoint',
            nms_mode='oks_nms')
        metric_coco.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in self.topdown_data_coco:
            metric_coco.process(data_batch, data_samples)

        eval_results = metric_coco.evaluate(size=len(self.topdown_data_coco))

        self.assertDictEqual(eval_results, self.target_coco)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test2.keypoints.json')))

        # case 3: score_mode='bbox_rle', nms_mode='soft_oks_nms'
        metric_coco = CocoWholeBodyMetric(
            ann_file=self.ann_file_coco,
            outfile_prefix=f'{self.tmp_dir.name}/test3',
            score_mode='bbox_rle',
            nms_mode='soft_oks_nms')
        metric_coco.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in self.topdown_data_coco:
            metric_coco.process(data_batch, data_samples)

        eval_results = metric_coco.evaluate(size=len(self.topdown_data_coco))

        self.assertDictEqual(eval_results, self.target_coco)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test3.keypoints.json')))

        # case 4: test without providing ann_file
        metric_coco = CocoWholeBodyMetric(
            outfile_prefix=f'{self.tmp_dir.name}/test4')
        metric_coco.dataset_meta = self.dataset_meta_coco
        # process samples
        for data_batch, data_samples in self.topdown_data_coco:
            metric_coco.process(data_batch, data_samples)
        eval_results = metric_coco.evaluate(size=len(self.topdown_data_coco))
        self.assertDictEqual(eval_results, self.target_coco)
        # test whether convert the annotation to COCO format
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test4.gt.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test4.keypoints.json')))
