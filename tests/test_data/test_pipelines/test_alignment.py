# Copyright (c) OpenMMLab. All rights reserved.
# Test alignment between pipeline2 and pipeline
# TODO: remove this unit test after 2.0 refactor is completed

from copy import deepcopy
from unittest import TestCase

import numpy as np
from mmcv.transforms import Compose

from mmpose.datasets import pipelines as pipelines1
from mmpose.datasets import pipelines2


class TestTopDownTransformAlignment(TestCase):

    def setUp(self):

        # data sample for pipeline 2.0
        data_info2 = dict(
            img=np.random.randint(0, 255, (480, 640, 3)).astype(np.uint8),
            img_shape=(480, 640, 3),
            bbox=np.array([[150, 150, 100, 100]], dtype=np.float32),
            bbox_rotation=np.array([15.], dtype=np.float32),
            bbox_score=np.ones(1, dtype=np.float32),
            keypoints=np.random.randint(180, 220,
                                        (1, 17, 2)).astype(np.float32),
            keypoints_visible=np.full((1, 17, 1), 1).astype(np.float32),
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            keypoint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

        # data sample for pipeline 1.0
        data_info1 = dict(
            img=data_info2['img'].copy(),
            bbox=data_info2['bbox'][0],
            rotation=data_info2['bbox_rotation'][0],
            joints_3d=np.zeros((17, 3), dtype=np.float32),
            joints_3d_visible=np.zeros((17, 3), dtype=np.float32),
            bbox_score=1)

        data_info1['joints_3d'][:, :2] = data_info2['keypoints'][0]
        data_info1['joints_3d_visible'][:, :2] = data_info2[
            'keypoints_visible'][0]

        data_info1['ann_info'] = dict(
            image_size=np.array([192, 256]),
            heatmap_size=np.array([48, 64]),
            num_joints=17,
            use_different_joint_weights=False,
            upper_body_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            lower_body_ids=[11, 12, 13, 14, 15, 16],
            flip_pairs=[[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                        [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                        [13, 14], [16, 15], [15, 16]],
            joint_weights=np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ]).astype(np.float32))

        self.data_info1 = data_info1
        self.data_info2 = data_info2

    def _check_output_aligned(self, results1, results2):
        """Check the outputs of pipeline1 and pipeline2 are aligned.

        Args:
            results1 (dict): The output of pipeline1
            results2 (dict): The output of pipeline2
        """
        # err = results2['img'] - results1['img']
        # print(np.where(err > 0))
        # print(err[np.where(err > 0)])
        self.assertTrue(
            np.allclose(results2['img'], results1['img']), '"img" not aligned')
        self.assertTrue(
            np.allclose(results2['bbox_center'], results1['center'][None]),
            '"bbox_center" is not aligned')
        self.assertTrue(
            np.allclose(results2['bbox_scale'],
                        results1['scale'][None] * 200.),
            '"bbox_scale" not aligned')  # pixel_std is 200.
        self.assertTrue(
            np.allclose(results2['keypoints'],
                        results1['joints_3d'][None, :, :2]),
            '"keypoints" not aligned')
        self.assertTrue(
            np.allclose(results2['keypoints_visible'],
                        results1['joints_3d_visible'][None, :, :2]),
            '"keypoints_visible" not aligned')
        self.assertTrue(
            np.allclose(
                results2['target_heatmap'], results1['target'], atol=1e-6),
            '"target_heatmap" not aligned')
        self.assertTrue(
            np.allclose(results2['target_weight'], results1['target_weight']),
            '"target_weight" not aligned')

    def test_alignment(self):
        # # msra
        # pipeline1 = Compose([
        #     pipelines1.TopDownGetBboxCenterScale(),
        #     pipelines1.TopDownAffine(use_udp=False),
        #     pipelines1.TopDownGenerateTarget(encoding='MSRA')
        # ])

        # pipeline2 = Compose([
        #     pipelines2.GetBboxCenterScale(),
        #     pipelines2.TopDownAffine(input_size=(192, 256), use_udp=False),
        #     pipelines2.TopDownGenerateHeatmap(
        #         heatmap_size=(48, 64),
        #         encoding='msra',
        #     ),
        # ])

        # results1 = pipeline1(deepcopy(self.data_info1))
        # results2 = pipeline2(deepcopy(self.data_info2))
        # self._check_output_aligned(results1, results2)

        # # msra + dark
        # pipeline1 = Compose([
        #     pipelines1.TopDownGetBboxCenterScale(),
        #     pipelines1.TopDownAffine(use_udp=False),
        #     pipelines1.TopDownGenerateTarget(
        #         encoding='MSRA',
        #         unbiased_encoding=True,
        #     )
        # ])

        # pipeline2 = Compose([
        #     pipelines2.GetBboxCenterScale(),
        #     pipelines2.TopDownAffine(input_size=(192, 256), use_udp=False),
        #     pipelines2.TopDownGenerateHeatmap(
        #         heatmap_size=(48, 64),
        #         encoding='msra',
        #         unbiased=True,
        #     ),
        # ])

        # results1 = pipeline1(deepcopy(self.data_info1))
        # results2 = pipeline2(deepcopy(self.data_info2))
        # self._check_output_aligned(results1, results2)

        # # megvii
        # pipeline1 = Compose([
        #     pipelines1.TopDownGetBboxCenterScale(),
        #     pipelines1.TopDownAffine(use_udp=False),
        #     pipelines1.TopDownGenerateTarget(encoding='Megvii')
        # ])

        # pipeline2 = Compose([
        #     pipelines2.GetBboxCenterScale(),
        #     pipelines2.TopDownAffine(input_size=(192, 256), use_udp=False),
        #     pipelines2.TopDownGenerateHeatmap(
        #         heatmap_size=(48, 64),
        #         encoding='megvii',
        #     ),
        # ])

        # results1 = pipeline1(deepcopy(self.data_info1))
        # results2 = pipeline2(deepcopy(self.data_info2))
        # self._check_output_aligned(results1, results2)

        # udp gaussian
        pipeline1 = Compose([
            pipelines1.TopDownGetBboxCenterScale(),
            pipelines1.TopDownAffine(use_udp=True),
            pipelines1.TopDownGenerateTarget(encoding='UDP')
        ])

        pipeline2 = Compose([
            pipelines2.GetBboxCenterScale(),
            pipelines2.TopDownAffine(input_size=(192, 256), use_udp=True),
            pipelines2.TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='udp',
            ),
        ])

        results1 = pipeline1(deepcopy(self.data_info1))
        results2 = pipeline2(deepcopy(self.data_info2))
        self._check_output_aligned(results1, results2)

        # udp combined
        pipeline1 = Compose([
            pipelines1.TopDownGetBboxCenterScale(),
            pipelines1.TopDownAffine(use_udp=True),
            pipelines1.TopDownGenerateTarget(
                encoding='UDP',
                target_type='CombinedTarget',
            )
        ])

        pipeline2 = Compose([
            pipelines2.GetBboxCenterScale(),
            pipelines2.TopDownAffine(input_size=(192, 256), use_udp=True),
            pipelines2.TopDownGenerateHeatmap(
                heatmap_size=(48, 64),
                encoding='udp',
                udp_combined_map=True,
            ),
        ])

        results1 = pipeline1(deepcopy(self.data_info1))
        results2 = pipeline2(deepcopy(self.data_info2))
        self._check_output_aligned(results1, results2)
