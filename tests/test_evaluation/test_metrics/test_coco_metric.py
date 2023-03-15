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
from mmpose.evaluation.metrics import CocoMetric


class TestCocoMetric(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.tmp_dir = tempfile.TemporaryDirectory()

        self.ann_file_coco = 'tests/data/coco/test_coco.json'
        meta_info_coco = dict(from_file='configs/_base_/datasets/coco.py')
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

        self.ann_file_crowdpose = 'tests/data/crowdpose/test_crowdpose.json'
        self.coco_crowdpose = COCO(self.ann_file_crowdpose)
        meta_info_crowdpose = dict(
            from_file='configs/_base_/datasets/crowdpose.py')
        self.dataset_meta_crowdpose = parse_pose_metainfo(meta_info_crowdpose)
        self.dataset_meta_crowdpose['CLASSES'] = self.coco_crowdpose.loadCats(
            self.coco_crowdpose.getCatIds())

        self.topdown_data_crowdpose = self._convert_ann_to_topdown_batch_data(
            self.ann_file_crowdpose)
        assert len(self.topdown_data_crowdpose) == 5
        self.bottomup_data_crowdpose = \
            self._convert_ann_to_bottomup_batch_data(self.ann_file_crowdpose)
        assert len(self.bottomup_data_crowdpose) == 2

        self.target_crowdpose = {
            'crowdpose/AP': 1.0,
            'crowdpose/AP .5': 1.0,
            'crowdpose/AP .75': 1.0,
            'crowdpose/AR': 1.0,
            'crowdpose/AR .5': 1.0,
            'crowdpose/AR .75': 1.0,
            'crowdpose/AP(E)': -1.0,
            'crowdpose/AP(M)': 1.0,
            'crowdpose/AP(H)': -1.0,
        }

        self.ann_file_ap10k = 'tests/data/ap10k/test_ap10k.json'
        self.coco_ap10k = COCO(self.ann_file_ap10k)
        meta_info_ap10k = dict(from_file='configs/_base_/datasets/ap10k.py')
        self.dataset_meta_ap10k = parse_pose_metainfo(meta_info_ap10k)
        self.dataset_meta_ap10k['CLASSES'] = self.coco_ap10k.loadCats(
            self.coco_ap10k.getCatIds())

        self.topdown_data_ap10k = self._convert_ann_to_topdown_batch_data(
            self.ann_file_ap10k)
        assert len(self.topdown_data_ap10k) == 2
        self.bottomup_data_ap10k = self._convert_ann_to_bottomup_batch_data(
            self.ann_file_ap10k)
        assert len(self.bottomup_data_ap10k) == 2

        self.target_ap10k = {
            'coco/AP': 1.0,
            'coco/AP .5': 1.0,
            'coco/AP .75': 1.0,
            'coco/AP (M)': -1.0,
            'coco/AP (L)': 1.0,
            'coco/AR': 1.0,
            'coco/AR .5': 1.0,
            'coco/AR .75': 1.0,
            'coco/AR (M)': -1.0,
            'coco/AR (L)': 1.0,
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
            keypoints = np.array(ann['keypoints']).reshape((1, -1, 3))

            gt_instances = {
                'bbox_scales': bbox_scales,
                'bbox_scores': np.ones((1, ), dtype=np.float32),
                'bboxes': bboxes,
            }
            pred_instances = {
                'keypoints': keypoints[..., :2],
                'keypoint_scores': keypoints[..., -1],
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

            # add crowd_index to data_sample if it is present in the image_info
            if 'crowdIndex' in imgid2info[ann['image_id']]:
                data_sample['crowd_index'] = imgid2info[
                    ann['image_id']]['crowdIndex']
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
            keypoints = np.array([ann['keypoints'] for ann in anns]).reshape(
                (len(anns), -1, 3))

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
            _ = CocoMetric(ann_file=self.ann_file_coco, score_mode='invalid')

        # test nms_mode option
        with self.assertRaisesRegex(ValueError, '`nms_mode` should be one of'):
            _ = CocoMetric(ann_file=self.ann_file_coco, nms_mode='invalid')

        # test format_only option
        with self.assertRaisesRegex(
                AssertionError,
                '`outfile_prefix` can not be None when `format_only` is True'):
            _ = CocoMetric(
                ann_file=self.ann_file_coco,
                format_only=True,
                outfile_prefix=None)

    def test_other_methods(self):
        """test other useful methods."""
        # test `_sort_and_unique_bboxes` method
        metric_coco = CocoMetric(
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
        metric_coco = CocoMetric(
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
            _ = CocoMetric(ann_file=tmp_ann_file, format_only=False)

    def test_bottomup_evaluate(self):
        """test bottomup-style COCO metric evaluation."""
        # case1: score_mode='bbox', nms_mode='none'
        metric_coco = CocoMetric(
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

    def test_topdown_alignment(self):
        """Test whether the output of CocoMetric and the original
        TopDownCocoDataset are the same."""
        topdown_data = []
        db = load(self.ann_file_coco)

        for ann in db['annotations']:
            w, h = ann['bbox'][2], ann['bbox'][3]
            bboxes = np.array(ann['bbox'], dtype=np.float32).reshape(-1, 4)
            bbox_scales = np.array([w * 1.25, h * 1.25]).reshape(-1, 2)

            keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
            keypoints[..., 0] = keypoints[..., 0] * 0.98
            keypoints[..., 1] = keypoints[..., 1] * 1.02
            keypoints[..., 2] = keypoints[..., 2] * 0.8

            gt_instances = {
                'bbox_scales': bbox_scales,
                'bbox_scores': np.ones((1, ), dtype=np.float32) * 0.98,
                'bboxes': bboxes,
            }
            pred_instances = {
                'keypoints': keypoints[..., :2],
                'keypoint_scores': keypoints[..., -1],
            }

            data = {'inputs': None}
            data_sample = {
                'id': ann['id'],
                'img_id': ann['image_id'],
                'gt_instances': gt_instances,
                'pred_instances': pred_instances
            }
            # batch size = 1
            data_batch = [data]
            data_samples = [data_sample]
            topdown_data.append((data_batch, data_samples))

        # case 1:
        # typical setting: score_mode='bbox_keypoint', nms_mode='oks_nms'
        metric_coco = CocoMetric(
            ann_file=self.ann_file_coco,
            outfile_prefix=f'{self.tmp_dir.name}/test_align1',
            score_mode='bbox_keypoint',
            nms_mode='oks_nms')
        metric_coco.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in topdown_data:
            metric_coco.process(data_batch, data_samples)

        eval_results = metric_coco.evaluate(size=len(topdown_data))

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
            osp.isfile(
                osp.join(self.tmp_dir.name, 'test_align1.keypoints.json')))

        # case 2: score_mode='bbox_rle', nms_mode='oks_nms'
        metric_coco = CocoMetric(
            ann_file=self.ann_file_coco,
            outfile_prefix=f'{self.tmp_dir.name}/test_align2',
            score_mode='bbox_rle',
            nms_mode='oks_nms')
        metric_coco.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in topdown_data:
            metric_coco.process(data_batch, data_samples)

        eval_results = metric_coco.evaluate(size=len(topdown_data))

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
            osp.isfile(
                osp.join(self.tmp_dir.name, 'test_align2.keypoints.json')))

        # case 3: score_mode='bbox_keypoint', nms_mode='soft_oks_nms'
        topdown_data = []
        anns = db['annotations']
        for i, ann in enumerate(anns):
            w, h = ann['bbox'][2], ann['bbox'][3]
            bboxes = np.array(ann['bbox'], dtype=np.float32).reshape(-1, 4)
            bbox_scales = np.array([w * 1.25, h * 1.25]).reshape(-1, 2)

            keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
            keypoints[..., 0] = keypoints[..., 0] * (1 - i / 100)
            keypoints[..., 1] = keypoints[..., 1] * (1 + i / 100)
            keypoints[..., 2] = keypoints[..., 2] * (1 - i / 100)

            gt_instances0 = {
                'bbox_scales': bbox_scales,
                'bbox_scores': np.ones((1, ), dtype=np.float32),
                'bboxes': bboxes,
            }
            pred_instances0 = {
                'keypoints': keypoints[..., :2],
                'keypoint_scores': keypoints[..., -1],
            }

            data0 = {'inputs': None}
            data_sample0 = {
                'id': ann['id'],
                'img_id': ann['image_id'],
                'gt_instances': gt_instances0,
                'pred_instances': pred_instances0
            }

            keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
            keypoints[..., 0] = keypoints[..., 0] * (1 + i / 100)
            keypoints[..., 1] = keypoints[..., 1] * (1 - i / 100)
            keypoints[..., 2] = keypoints[..., 2] * (1 - 2 * i / 100)

            gt_instances1 = {
                'bbox_scales': bbox_scales,
                'bboxes': bboxes,
                'bbox_scores': np.ones(
                    (1, ), dtype=np.float32) * (1 - 2 * i / 100)
            }
            pred_instances1 = {
                'keypoints': keypoints[..., :2],
                'keypoint_scores': keypoints[..., -1],
            }

            data1 = {'inputs': None}
            data_sample1 = {
                'id': ann['id'] + 1,
                'img_id': ann['image_id'],
                'gt_instances': gt_instances1,
                'pred_instances': pred_instances1
            }

            # batch size = 2
            data_batch = [data0, data1]
            data_samples = [data_sample0, data_sample1]
            topdown_data.append((data_batch, data_samples))

        metric_coco = CocoMetric(
            ann_file=self.ann_file_coco,
            outfile_prefix=f'{self.tmp_dir.name}/test_align3',
            score_mode='bbox_keypoint',
            keypoint_score_thr=0.2,
            nms_thr=0.9,
            nms_mode='soft_oks_nms')
        metric_coco.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in topdown_data:
            metric_coco.process(data_batch, data_samples)

        eval_results = metric_coco.evaluate(size=len(topdown_data) * 2)

        target = {
            'coco/AP': 0.17073707370737073,
            'coco/AP .5': 0.25055005500550054,
            'coco/AP .75': 0.10671067106710669,
            'coco/AP (M)': 0.0,
            'coco/AP (L)': 0.29315181518151806,
            'coco/AR': 0.2416666666666666,
            'coco/AR .5': 0.3333333333333333,
            'coco/AR .75': 0.16666666666666666,
            'coco/AR (M)': 0.0,
            'coco/AR (L)': 0.41428571428571426,
        }

        for key in eval_results.keys():
            self.assertAlmostEqual(eval_results[key], target[key])

        self.assertTrue(
            osp.isfile(
                osp.join(self.tmp_dir.name, 'test_align3.keypoints.json')))

    def test_topdown_evaluate(self):
        """test topdown-style COCO metric evaluation."""
        # case 1: score_mode='bbox', nms_mode='none'
        metric_coco = CocoMetric(
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
        metric_coco = CocoMetric(
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
        metric_coco = CocoMetric(
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
        metric_coco = CocoMetric(outfile_prefix=f'{self.tmp_dir.name}/test4')
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

        # case 5: test Crowdpose dataset
        metric_crowdpose = CocoMetric(
            ann_file=self.ann_file_crowdpose,
            outfile_prefix=f'{self.tmp_dir.name}/test5',
            use_area=False,
            iou_type='keypoints_crowd',
            prefix='crowdpose')
        metric_crowdpose.dataset_meta = self.dataset_meta_crowdpose
        # process samples
        for data_batch, data_samples in self.topdown_data_crowdpose:
            metric_crowdpose.process(data_batch, data_samples)
        eval_results = metric_crowdpose.evaluate(
            size=len(self.topdown_data_crowdpose))
        self.assertDictEqual(eval_results, self.target_crowdpose)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test5.keypoints.json')))

        # case 6: test Crowdpose dataset + without ann_file
        metric_crowdpose = CocoMetric(
            outfile_prefix=f'{self.tmp_dir.name}/test6',
            use_area=False,
            iou_type='keypoints_crowd',
            prefix='crowdpose')
        metric_crowdpose.dataset_meta = self.dataset_meta_crowdpose
        # process samples
        for data_batch, data_samples in self.topdown_data_crowdpose:
            metric_crowdpose.process(data_batch, data_samples)
        eval_results = metric_crowdpose.evaluate(
            size=len(self.topdown_data_crowdpose))
        self.assertDictEqual(eval_results, self.target_crowdpose)
        # test whether convert the annotation to COCO format
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test6.gt.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test6.keypoints.json')))

        # case 7: test AP10k dataset
        metric_ap10k = CocoMetric(
            ann_file=self.ann_file_ap10k,
            outfile_prefix=f'{self.tmp_dir.name}/test7')
        metric_ap10k.dataset_meta = self.dataset_meta_ap10k
        # process samples
        for data_batch, data_samples in self.topdown_data_ap10k:
            metric_ap10k.process(data_batch, data_samples)
        eval_results = metric_ap10k.evaluate(size=len(self.topdown_data_ap10k))
        for key in self.target_ap10k:
            self.assertAlmostEqual(eval_results[key], self.target_ap10k[key])
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test7.keypoints.json')))

        # case 8: test Crowdpose dataset + without ann_file
        metric_ap10k = CocoMetric(outfile_prefix=f'{self.tmp_dir.name}/test8')
        metric_ap10k.dataset_meta = self.dataset_meta_ap10k
        # process samples
        for data_batch, data_samples in self.topdown_data_ap10k:
            metric_ap10k.process(data_batch, data_samples)
        eval_results = metric_ap10k.evaluate(size=len(self.topdown_data_ap10k))
        for key in self.target_ap10k:
            self.assertAlmostEqual(eval_results[key], self.target_ap10k[key])
        # test whether convert the annotation to COCO format
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test8.gt.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test8.keypoints.json')))
