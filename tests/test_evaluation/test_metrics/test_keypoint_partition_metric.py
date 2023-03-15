# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
from collections import defaultdict
from unittest import TestCase

import numpy as np
from mmengine.fileio import load
from mmengine.structures import InstanceData
from xtcocotools.coco import COCO

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.evaluation.metrics import KeypointPartitionMetric


class TestKeypointPartitionMetricWrappingCocoMetric(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.tmp_dir = tempfile.TemporaryDirectory()

        self.ann_file_coco = \
            'tests/data/coco/test_keypoint_partition_metric.json'
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
        """
        The target results were obtained from CocoWholebodyMetric with
        score_mode='bbox' and nms_mode='none'. We cannot compare other
        combinations of score_mode and nms_mode because CocoWholebodyMetric
        calculates scores and nms using all keypoints while
        KeypointPartitionMetric calculates scores and nms part by part.
        As long as this case is tested correct, the other cases should be
        correct.
        """
        self.target_bbox_none = {
            'body/coco/AP': 0.749,
            'body/coco/AR': 0.800,
            'foot/coco/AP': 0.840,
            'foot/coco/AR': 0.850,
            'face/coco/AP': 0.051,
            'face/coco/AR': 0.050,
            'left_hand/coco/AP': 0.283,
            'left_hand/coco/AR': 0.300,
            'right_hand/coco/AP': 0.383,
            'right_hand/coco/AR': 0.380,
            'all/coco/AP': 0.284,
            'all/coco/AR': 0.450,
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
            _keypoints = np.array(ann['keypoints']).reshape((1, -1, 3))

            gt_instances = {
                'bbox_scales': bbox_scales,
                'bbox_scores': np.ones((1, ), dtype=np.float32),
                'bboxes': bboxes,
                'keypoints': _keypoints[..., :2],
                'keypoints_visible': _keypoints[..., 2:3]
            }

            # fake predictions
            keypoints = np.zeros_like(_keypoints)
            keypoints[..., 0] = _keypoints[..., 0] * 0.99
            keypoints[..., 1] = _keypoints[..., 1] * 1.02
            keypoints[..., 2] = _keypoints[..., 2] * 0.8

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
            _keypoints = np.array([ann['keypoints'] for ann in anns]).reshape(
                (len(anns), -1, 3))

            gt_instances = {
                'bbox_scores': np.ones((len(anns)), dtype=np.float32),
                'keypoints': _keypoints[..., :2],
                'keypoints_visible': _keypoints[..., 2:3]
            }

            # fake predictions
            keypoints = np.zeros_like(_keypoints)
            keypoints[..., 0] = _keypoints[..., 0] * 0.99
            keypoints[..., 1] = _keypoints[..., 1] * 1.02
            keypoints[..., 2] = _keypoints[..., 2] * 0.8

            pred_instances = {
                'keypoints': keypoints[..., :2],
                'keypoint_scores': keypoints[..., -1],
            }

            data = {'inputs': None}
            data_sample = {
                'id': [ann['id'] for ann in anns],
                'img_id': img_id,
                'gt_instances': gt_instances,
                'pred_instances': pred_instances,
                # dummy image_shape for testing
                'ori_shape': [640, 480],
                'raw_ann_info': copy.deepcopy(anns),
            }

            # batch size = 1
            data_batch = [data]
            data_samples = [data_sample]
            bottomup_data.append((data_batch, data_samples))
        return bottomup_data

    def _assert_outfiles(self, prefix):
        for part in ['body', 'foot', 'face', 'left_hand', 'right_hand', 'all']:
            self.assertTrue(
                osp.isfile(
                    osp.join(self.tmp_dir.name,
                             f'{prefix}.{part}.keypoints.json')))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        """test metric init method."""
        # test wrong metric type
        with self.assertRaisesRegex(
                ValueError, 'Metrics supported by KeypointPartitionMetric'):
            _ = KeypointPartitionMetric(
                metric=dict(type='Metric'), partitions=dict(all=range(133)))

        # test ann_file arg warning
        with self.assertWarnsRegex(UserWarning,
                                   'does not support the ann_file argument'):
            _ = KeypointPartitionMetric(
                metric=dict(type='CocoMetric', ann_file=''),
                partitions=dict(all=range(133)))

        # test score_mode arg warning
        with self.assertWarnsRegex(UserWarning, "if score_mode is not 'bbox'"):
            _ = KeypointPartitionMetric(
                metric=dict(type='CocoMetric'),
                partitions=dict(all=range(133)))

        # test nms arg warning
        with self.assertWarnsRegex(UserWarning, 'oks_nms and soft_oks_nms'):
            _ = KeypointPartitionMetric(
                metric=dict(type='CocoMetric'),
                partitions=dict(all=range(133)))

        # test partitions
        with self.assertRaisesRegex(AssertionError, 'at least one partition'):
            _ = KeypointPartitionMetric(
                metric=dict(type='CocoMetric'), partitions=dict())

        with self.assertRaisesRegex(AssertionError, 'should be a sequence'):
            _ = KeypointPartitionMetric(
                metric=dict(type='CocoMetric'), partitions=dict(all={}))

        with self.assertRaisesRegex(AssertionError, 'at least one element'):
            _ = KeypointPartitionMetric(
                metric=dict(type='CocoMetric'), partitions=dict(all=[]))

    def test_bottomup_evaluate(self):
        """test bottomup-style COCO metric evaluation."""
        # case1: score_mode='bbox', nms_mode='none'
        metric = KeypointPartitionMetric(
            metric=dict(
                type='CocoMetric',
                outfile_prefix=f'{self.tmp_dir.name}/test_bottomup',
                score_mode='bbox',
                nms_mode='none'),
            partitions=dict(
                body=range(17),
                foot=range(17, 23),
                face=range(23, 91),
                left_hand=range(91, 112),
                right_hand=range(112, 133),
                all=range(133)))
        metric.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in self.bottomup_data_coco:
            metric.process(data_batch, data_samples)

        eval_results = metric.evaluate(size=len(self.bottomup_data_coco))
        for key in self.target_bbox_none.keys():
            self.assertAlmostEqual(
                eval_results[key], self.target_bbox_none[key], places=3)
        self._assert_outfiles('test_bottomup')

    def test_topdown_evaluate(self):
        """test topdown-style COCO metric evaluation."""
        # case 1: score_mode='bbox', nms_mode='none'
        metric = KeypointPartitionMetric(
            metric=dict(
                type='CocoMetric',
                outfile_prefix=f'{self.tmp_dir.name}/test_topdown1',
                score_mode='bbox',
                nms_mode='none'),
            partitions=dict(
                body=range(17),
                foot=range(17, 23),
                face=range(23, 91),
                left_hand=range(91, 112),
                right_hand=range(112, 133),
                all=range(133)))
        metric.dataset_meta = self.dataset_meta_coco

        # process samples
        for data_batch, data_samples in self.topdown_data_coco:
            metric.process(data_batch, data_samples)

        eval_results = metric.evaluate(size=len(self.topdown_data_coco))
        for key in self.target_bbox_none.keys():
            self.assertAlmostEqual(
                eval_results[key], self.target_bbox_none[key], places=3)
        self._assert_outfiles('test_topdown1')


class TestKeypointPartitionMetricWrappingPCKAccuracy(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.batch_size = 8
        num_keypoints = 24
        self.data_batch = []
        self.data_samples = []

        for i in range(self.batch_size):
            gt_instances = InstanceData()
            keypoints = np.zeros((1, num_keypoints, 2))
            for j in range(num_keypoints):
                keypoints[0, j] = [0.5 * i * j, 0.5 * i * j]
            gt_instances.keypoints = keypoints
            gt_instances.keypoints_visible = np.ones(
                (1, num_keypoints, 1)).astype(bool)
            gt_instances.keypoints_visible[0, (2 * i) % 8, 0] = False
            gt_instances.bboxes = np.array([[0.1, 0.2, 0.3, 0.4]]) * 20 * i
            gt_instances.head_size = np.array([[0.1]]) * 10 * i

            pred_instances = InstanceData()
            # fake predictions
            _keypoints = np.zeros_like(keypoints)
            _keypoints[0, :, 0] = keypoints[0, :, 0] * 0.95
            _keypoints[0, :, 1] = keypoints[0, :, 1] * 1.05
            pred_instances.keypoints = _keypoints

            data = {'inputs': None}
            data_sample = {
                'gt_instances': gt_instances.to_dict(),
                'pred_instances': pred_instances.to_dict(),
            }

            self.data_batch.append(data)
            self.data_samples.append(data_sample)

    def test_init(self):
        # test norm_item arg warning
        with self.assertWarnsRegex(UserWarning,
                                   'norm_item torso is used in JhmdbDataset'):
            _ = KeypointPartitionMetric(
                metric=dict(
                    type='PCKAccuracy', thr=0.05, norm_item=['bbox', 'torso']),
                partitions=dict(all=range(133)))

    def test_evaluate(self):
        """test PCK accuracy evaluation metric."""
        # test normalized by 'bbox'
        pck_metric = KeypointPartitionMetric(
            metric=dict(type='PCKAccuracy', thr=0.5, norm_item='bbox'),
            partitions=dict(
                p1=range(10),
                p2=range(10, 24),
                all=range(24),
            ))
        pck_metric.process(self.data_batch, self.data_samples)
        pck = pck_metric.evaluate(self.batch_size)
        target = {'p1/PCK': 1.0, 'p2/PCK': 1.0, 'all/PCK': 1.0}
        self.assertDictEqual(pck, target)

        # test normalized by 'head_size'
        pckh_metric = KeypointPartitionMetric(
            metric=dict(type='PCKAccuracy', thr=0.3, norm_item='head'),
            partitions=dict(
                p1=range(10),
                p2=range(10, 24),
                all=range(24),
            ))
        pckh_metric.process(self.data_batch, self.data_samples)
        pckh = pckh_metric.evaluate(self.batch_size)
        target = {'p1/PCKh': 0.9, 'p2/PCKh': 0.0, 'all/PCKh': 0.375}
        self.assertDictEqual(pckh, target)

        # test normalized by 'torso_size'
        tpck_metric = KeypointPartitionMetric(
            metric=dict(
                type='PCKAccuracy', thr=0.05, norm_item=['bbox', 'torso']),
            partitions=dict(
                p1=range(10),
                p2=range(10, 24),
                all=range(24),
            ))
        tpck_metric.process(self.data_batch, self.data_samples)
        tpck = tpck_metric.evaluate(self.batch_size)
        self.assertIsInstance(tpck, dict)
        target = {
            'p1/PCK': 0.6,
            'p1/tPCK': 0.11428571428571428,
            'p2/PCK': 0.0,
            'p2/tPCK': 0.0,
            'all/PCK': 0.25,
            'all/tPCK': 0.047619047619047616
        }
        self.assertDictEqual(tpck, target)


class TestKeypointPartitionMetricWrappingAUCandEPE(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        output = np.zeros((1, 5, 2))
        target = np.zeros((1, 5, 2))
        # first channel
        output[0, 0] = [10, 4]
        target[0, 0] = [10, 0]
        # second channel
        output[0, 1] = [10, 18]
        target[0, 1] = [10, 10]
        # third channel
        output[0, 2] = [0, 0]
        target[0, 2] = [0, -1]
        # fourth channel
        output[0, 3] = [40, 40]
        target[0, 3] = [30, 30]
        # fifth channel
        output[0, 4] = [20, 10]
        target[0, 4] = [0, 10]

        gt_instances = InstanceData()
        gt_instances.keypoints = target
        gt_instances.keypoints_visible = np.array(
            [[True, True, False, True, True]])

        pred_instances = InstanceData()
        pred_instances.keypoints = output

        data = {'inputs': None}
        data_sample = {
            'gt_instances': gt_instances.to_dict(),
            'pred_instances': pred_instances.to_dict()
        }

        self.data_batch = [data]
        self.data_samples = [data_sample]

    def test_auc_evaluate(self):
        """test AUC evaluation metric."""
        auc_metric = KeypointPartitionMetric(
            metric=dict(type='AUC', norm_factor=20, num_thrs=4),
            partitions=dict(
                p1=range(3),
                p2=range(3, 5),
                all=range(5),
            ))
        auc_metric.process(self.data_batch, self.data_samples)
        auc = auc_metric.evaluate(1)
        target = {'p1/AUC': 0.625, 'p2/AUC': 0.125, 'all/AUC': 0.375}
        self.assertDictEqual(auc, target)

    def test_epe_evaluate(self):
        """test EPE evaluation metric."""
        epe_metric = KeypointPartitionMetric(
            metric=dict(type='EPE', ),
            partitions=dict(
                p1=range(3),
                p2=range(3, 5),
                all=range(5),
            ))
        epe_metric.process(self.data_batch, self.data_samples)
        epe = epe_metric.evaluate(1)
        target = {
            'p1/EPE': 6.0,
            'p2/EPE': 17.071067810058594,
            'all/EPE': 11.535533905029297
        }
        self.assertDictEqual(epe, target)


class TestKeypointPartitionMetricWrappingNME(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.batch_size = 4
        num_keypoints = 19
        self.data_batch = []
        self.data_samples = []

        for i in range(self.batch_size):
            gt_instances = InstanceData()
            keypoints = np.zeros((1, num_keypoints, 2))
            for j in range(num_keypoints):
                keypoints[0, j] = [0.5 * i * j, 0.5 * i * j]
            gt_instances.keypoints = keypoints
            gt_instances.keypoints_visible = np.ones(
                (1, num_keypoints, 1)).astype(bool)
            gt_instances.keypoints_visible[0, (2 * i) % self.batch_size,
                                           0] = False
            gt_instances['box_size'] = np.array([[0.1]]) * 10 * i

            pred_instances = InstanceData()
            # fake predictions
            _keypoints = np.zeros_like(keypoints)
            _keypoints[0, :, 0] = keypoints[0, :, 0] * 0.95
            _keypoints[0, :, 1] = keypoints[0, :, 1] * 1.05
            pred_instances.keypoints = _keypoints

            data = {'inputs': None}
            data_sample = {
                'gt_instances': gt_instances.to_dict(),
                'pred_instances': pred_instances.to_dict(),
            }

            self.data_batch.append(data)
            self.data_samples.append(data_sample)

    def test_init(self):
        # test norm_mode arg missing
        with self.assertRaisesRegex(AssertionError, 'Missing norm_mode'):
            _ = KeypointPartitionMetric(
                metric=dict(type='NME', ), partitions=dict(all=range(133)))

        # test norm_mode = keypoint_distance
        with self.assertRaisesRegex(ValueError,
                                    "NME norm_mode 'keypoint_distance'"):
            _ = KeypointPartitionMetric(
                metric=dict(type='NME', norm_mode='keypoint_distance'),
                partitions=dict(all=range(133)))

    def test_nme_evaluate(self):
        """test NME evaluation metric."""
        # test when norm_mode = 'use_norm_item'
        # test norm_item = 'box_size' like in `AFLWDataset`
        nme_metric = KeypointPartitionMetric(
            metric=dict(
                type='NME', norm_mode='use_norm_item', norm_item='box_size'),
            partitions=dict(
                p1=range(10),
                p2=range(10, 19),
                all=range(19),
            ))
        nme_metric.process(self.data_batch, self.data_samples)
        nme = nme_metric.evaluate(4)
        target = {
            'p1/NME': 0.1715388651247378,
            'p2/NME': 0.4949747721354167,
            'all/NME': 0.333256827460395
        }
        self.assertDictEqual(nme, target)
