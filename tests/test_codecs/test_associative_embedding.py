# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from unittest import TestCase

import numpy as np
import torch
from munkres import Munkres

from mmpose.codecs import AssociativeEmbedding
from mmpose.registry import KEYPOINT_CODECS
from mmpose.testing import get_coco_sample


class TestAssociativeEmbedding(TestCase):

    def setUp(self) -> None:
        self.decode_keypoint_order = [
            0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16
        ]

    def test_build(self):
        cfg = dict(
            type='AssociativeEmbedding',
            input_size=(256, 256),
            heatmap_size=(64, 64),
            use_udp=False,
            decode_keypoint_order=self.decode_keypoint_order,
        )
        codec = KEYPOINT_CODECS.build(cfg)
        self.assertIsInstance(codec, AssociativeEmbedding)

    def test_encode(self):
        data = get_coco_sample(img_shape=(256, 256), num_instances=1)

        # w/o UDP
        codec = AssociativeEmbedding(
            input_size=(256, 256),
            heatmap_size=(64, 64),
            use_udp=False,
            decode_keypoint_order=self.decode_keypoint_order)

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])

        heatmaps = encoded['heatmaps']
        keypoint_indices = encoded['keypoint_indices']
        keypoint_weights = encoded['keypoint_weights']

        self.assertEqual(heatmaps.shape, (17, 64, 64))
        self.assertEqual(keypoint_indices.shape, (1, 17, 2))
        self.assertEqual(keypoint_weights.shape, (1, 17))

        for k in range(heatmaps.shape[0]):
            index_expected = np.argmax(heatmaps[k])
            index_encoded = keypoint_indices[0, k, 0]
            self.assertEqual(index_expected, index_encoded)

        # w/ UDP
        codec = AssociativeEmbedding(
            input_size=(256, 256),
            heatmap_size=(64, 64),
            use_udp=True,
            decode_keypoint_order=self.decode_keypoint_order)

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])

        heatmaps = encoded['heatmaps']
        keypoint_indices = encoded['keypoint_indices']
        keypoint_weights = encoded['keypoint_weights']

        self.assertEqual(heatmaps.shape, (17, 64, 64))
        self.assertEqual(keypoint_indices.shape, (1, 17, 2))
        self.assertEqual(keypoint_weights.shape, (1, 17))

        for k in range(heatmaps.shape[0]):
            index_expected = np.argmax(heatmaps[k])
            index_encoded = keypoint_indices[0, k, 0]
            self.assertEqual(index_expected, index_encoded)

    def _get_tags(self,
                  heatmaps,
                  keypoint_indices,
                  tag_per_keypoint: bool,
                  tag_dim: int = 1):

        K, H, W = heatmaps.shape
        N = keypoint_indices.shape[0]

        if tag_per_keypoint:
            tags = np.zeros((K * tag_dim, H, W), dtype=np.float32)
        else:
            tags = np.zeros((tag_dim, H, W), dtype=np.float32)

        for n, k in product(range(N), range(K)):
            y, x = np.unravel_index(keypoint_indices[n, k, 0], (H, W))
            if tag_per_keypoint:
                tags[k::K, y, x] = n
            else:
                tags[:, y, x] = n

        return tags

    def _sort_preds(self, keypoints_pred, scores_pred, keypoints_gt):
        """Sort multi-instance predictions to best match the ground-truth.

        Args:
            keypoints_pred (np.ndarray): predictions in shape (N, K, D)
            scores (np.ndarray): predictions in shape (N, K)
            keypoints_gt (np.ndarray): ground-truth in shape (N, K, D)

        Returns:
            np.ndarray: Sorted predictions
        """
        assert keypoints_gt.shape == keypoints_pred.shape
        costs = np.linalg.norm(
            keypoints_gt[None] - keypoints_pred[:, None], ord=2,
            axis=3).mean(axis=2)
        match = Munkres().compute(costs)
        keypoints_pred_sorted = np.zeros_like(keypoints_pred)
        scores_pred_sorted = np.zeros_like(scores_pred)
        for i, j in match:
            keypoints_pred_sorted[i] = keypoints_pred[j]
            scores_pred_sorted[i] = scores_pred[j]

        return keypoints_pred_sorted, scores_pred_sorted

    def test_decode(self):
        data = get_coco_sample(
            img_shape=(256, 256), num_instances=2, non_occlusion=True)

        # w/o UDP
        codec = AssociativeEmbedding(
            input_size=(256, 256),
            heatmap_size=(64, 64),
            use_udp=False,
            decode_keypoint_order=self.decode_keypoint_order)

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])

        heatmaps = encoded['heatmaps']
        keypoint_indices = encoded['keypoint_indices']

        tags = self._get_tags(
            heatmaps, keypoint_indices, tag_per_keypoint=True)

        # to Tensor
        batch_heatmaps = torch.from_numpy(heatmaps[None])
        batch_tags = torch.from_numpy(tags[None])

        batch_keypoints, batch_keypoint_scores = codec.batch_decode(
            batch_heatmaps, batch_tags)

        self.assertIsInstance(batch_keypoints, list)
        self.assertIsInstance(batch_keypoint_scores, list)
        self.assertEqual(len(batch_keypoints), 1)
        self.assertEqual(len(batch_keypoint_scores), 1)

        keypoints, scores = self._sort_preds(batch_keypoints[0],
                                             batch_keypoint_scores[0],
                                             data['keypoints'])

        self.assertIsInstance(keypoints, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(keypoints.shape, (2, 17, 2))
        self.assertEqual(scores.shape, (2, 17))

        self.assertTrue(np.allclose(keypoints, data['keypoints'], atol=4.0))

        # w/o UDP, tag_imd=2
        codec = AssociativeEmbedding(
            input_size=(256, 256),
            heatmap_size=(64, 64),
            use_udp=False,
            decode_keypoint_order=self.decode_keypoint_order)

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])

        heatmaps = encoded['heatmaps']
        keypoint_indices = encoded['keypoint_indices']

        tags = self._get_tags(
            heatmaps, keypoint_indices, tag_per_keypoint=True, tag_dim=2)

        # to Tensor
        batch_heatmaps = torch.from_numpy(heatmaps[None])
        batch_tags = torch.from_numpy(tags[None])

        batch_keypoints, batch_keypoint_scores = codec.batch_decode(
            batch_heatmaps, batch_tags)

        self.assertIsInstance(batch_keypoints, list)
        self.assertIsInstance(batch_keypoint_scores, list)
        self.assertEqual(len(batch_keypoints), 1)
        self.assertEqual(len(batch_keypoint_scores), 1)

        keypoints, scores = self._sort_preds(batch_keypoints[0],
                                             batch_keypoint_scores[0],
                                             data['keypoints'])

        self.assertIsInstance(keypoints, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(keypoints.shape, (2, 17, 2))
        self.assertEqual(scores.shape, (2, 17))

        self.assertTrue(np.allclose(keypoints, data['keypoints'], atol=4.0))

        # w/ UDP
        codec = AssociativeEmbedding(
            input_size=(256, 256),
            heatmap_size=(64, 64),
            use_udp=True,
            decode_keypoint_order=self.decode_keypoint_order)

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])

        heatmaps = encoded['heatmaps']
        keypoint_indices = encoded['keypoint_indices']

        tags = self._get_tags(
            heatmaps, keypoint_indices, tag_per_keypoint=True)

        # to Tensor
        batch_heatmaps = torch.from_numpy(heatmaps[None])
        batch_tags = torch.from_numpy(tags[None])

        batch_keypoints, batch_keypoint_scores = codec.batch_decode(
            batch_heatmaps, batch_tags)

        self.assertIsInstance(batch_keypoints, list)
        self.assertIsInstance(batch_keypoint_scores, list)
        self.assertEqual(len(batch_keypoints), 1)
        self.assertEqual(len(batch_keypoint_scores), 1)

        keypoints, scores = self._sort_preds(batch_keypoints[0],
                                             batch_keypoint_scores[0],
                                             data['keypoints'])

        self.assertIsInstance(keypoints, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(keypoints.shape, (2, 17, 2))
        self.assertEqual(scores.shape, (2, 17))

        self.assertTrue(np.allclose(keypoints, data['keypoints'], atol=4.0))
