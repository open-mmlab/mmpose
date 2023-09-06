# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.evaluation.evaluators import MultiDatasetEvaluator
from mmpose.testing import get_coco_sample
from mmpose.utils import register_all_modules


class TestMultiDatasetEvaluator(TestCase):

    def setUp(self) -> None:
        register_all_modules()

        aic_to_coco_converter = dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[
                (0, 6),
                (1, 8),
                (2, 10),
                (3, 5),
                (4, 7),
                (5, 9),
                (6, 12),
                (7, 14),
                (8, 16),
                (9, 11),
                (10, 13),
                (11, 15),
            ])

        # val datasets
        dataset_coco_val = dict(
            type='CocoDataset',
            data_root='data/coco',
            test_mode=True,
        )

        dataset_aic_val = dict(
            type='AicDataset',
            data_root='data/aic/',
            test_mode=True,
        )

        self.datasets = [dataset_coco_val, dataset_aic_val]

        self.metrics = [
            dict(type='CocoMetric', ann_file='tests/data/coco/test_coco.json'),
            dict(
                type='CocoMetric',
                ann_file='tests/data/aic/test_aic.json',
                use_area=False,
                gt_converter=aic_to_coco_converter,
                prefix='aic')
        ]

        data_sample1 = get_coco_sample(
            img_shape=(240, 320), num_instances=2, with_bbox_cs=False)
        data_sample1['dataset_name'] = 'coco'
        data_sample1['id'] = 0
        data_sample1['img_id'] = 100
        data_sample1['gt_instances'] = dict(bbox_scores=np.ones(2), )
        data_sample1['pred_instances'] = dict(
            keypoints=data_sample1['keypoints'],
            keypoint_scores=data_sample1['keypoints_visible'],
        )
        imgs1 = data_sample1.pop('img')

        data_sample2 = get_coco_sample(
            img_shape=(240, 320), num_instances=3, with_bbox_cs=False)
        data_sample2['dataset_name'] = 'aic'
        data_sample2['id'] = 1
        data_sample2['img_id'] = 200
        data_sample2['gt_instances'] = dict(bbox_scores=np.ones(3), )
        data_sample2['pred_instances'] = dict(
            keypoints=data_sample2['keypoints'],
            keypoint_scores=data_sample2['keypoints_visible'],
        )
        imgs2 = data_sample2.pop('img')

        self.data_batch = dict(
            inputs=[imgs1, imgs2], data_samples=[data_sample1, data_sample2])
        self.data_samples = [data_sample1, data_sample2]

    def test_init(self):
        evaluator = MultiDatasetEvaluator(self.metrics, self.datasets)
        self.assertIn('metrics_dict', dir(evaluator))
        self.assertEqual(len(evaluator.metrics_dict), 2)

        with self.assertRaises(AssertionError):
            evaluator = MultiDatasetEvaluator(self.metrics, self.datasets[:1])

    def test_process(self):
        evaluator = MultiDatasetEvaluator(self.metrics, self.datasets)
        evaluator.dataset_meta = dict(dataset_name='default')
        evaluator.process(self.data_samples, self.data_batch)

        for metric in evaluator.metrics:
            self.assertGreater(len(metric.results), 0)
