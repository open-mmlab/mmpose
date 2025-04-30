# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from itertools import product
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_tuple_of

from mmpose.codecs import AssociativeEmbedding  # noqa
from mmpose.models.heads import AssociativeEmbeddingHead
from mmpose.registry import KEYPOINT_CODECS
from mmpose.testing._utils import get_packed_inputs


class TestAssociativeEmbeddingHead(TestCase):

    def _get_tags(self, heatmaps, keypoint_indices, tag_per_keypoint: bool):

        K, H, W = heatmaps.shape
        N = keypoint_indices.shape[0]

        if tag_per_keypoint:
            tags = np.zeros((K, H, W), dtype=np.float32)
        else:
            tags = np.zeros((1, H, W), dtype=np.float32)

        for n, k in product(range(N), range(K)):
            y, x = np.unravel_index(keypoint_indices[n, k, 0], (H, W))
            if tag_per_keypoint:
                tags[k, y, x] = n
            else:
                tags[0, y, x] = n

        return tags

    def test_forward(self):

        head = AssociativeEmbeddingHead(
            in_channels=32,
            num_keypoints=17,
            tag_dim=1,
            tag_per_keypoint=True,
            deconv_out_channels=None)

        feats = [torch.rand(1, 32, 64, 64)]
        output = head.forward(feats)  # should be (heatmaps, tags)
        self.assertTrue(is_tuple_of(output, torch.Tensor))
        self.assertEqual(output[0].shape, (1, 17, 64, 64))
        self.assertEqual(output[1].shape, (1, 17, 64, 64))

    def test_predict(self):

        codec_cfg = dict(
            type='AssociativeEmbedding',
            input_size=(256, 256),
            heatmap_size=(64, 64),
            use_udp=False,
            decode_keypoint_order=[
                0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16
            ])

        # get test data
        codec = KEYPOINT_CODECS.build(codec_cfg)
        batch_data_samples = get_packed_inputs(
            1,
            input_size=(256, 256),
            heatmap_size=(64, 64),
            img_shape=(256, 256))['data_samples']

        keypoints = batch_data_samples[0].gt_instances['keypoints']
        keypoints_visible = batch_data_samples[0].gt_instances[
            'keypoints_visible']

        encoded = codec.encode(keypoints, keypoints_visible)
        heatmaps = encoded['heatmaps']
        keypoint_indices = encoded['keypoint_indices']

        tags = self._get_tags(
            heatmaps, keypoint_indices, tag_per_keypoint=True)

        dummy_feat = np.concatenate((heatmaps, tags), axis=0)
        feats = [torch.from_numpy(dummy_feat)[None]]

        head = AssociativeEmbeddingHead(
            in_channels=34,
            num_keypoints=17,
            tag_dim=1,
            tag_per_keypoint=True,
            deconv_out_channels=None,
            final_layer=None,
            decoder=codec_cfg)

        preds = head.predict(feats, batch_data_samples)
        self.assertTrue(np.allclose(preds[0].keypoints, keypoints, atol=4.0))

    def test_loss(self):

        codec_cfg = dict(
            type='AssociativeEmbedding',
            input_size=(256, 256),
            heatmap_size=(64, 64),
            use_udp=False,
            decode_keypoint_order=[
                0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16
            ])

        # get test data
        codec = KEYPOINT_CODECS.build(codec_cfg)

        batch_data_samples = get_packed_inputs(
            1,
            input_size=(256, 256),
            heatmap_size=(64, 64),
            img_shape=(256, 256))['data_samples']

        keypoints = batch_data_samples[0].gt_instances['keypoints']
        keypoints_visible = batch_data_samples[0].gt_instances[
            'keypoints_visible']
        encoded = codec.encode(keypoints, keypoints_visible)
        heatmaps = encoded['heatmaps']
        keypoint_indices = encoded['keypoint_indices']
        keypoint_weights = encoded['keypoint_weights']

        heatmap_mask = np.ones((1, ) + heatmaps.shape[1:], dtype=np.float32)
        batch_data_samples[0].gt_fields = PixelData(
            heatmaps=heatmaps, heatmap_mask=heatmap_mask).to_tensor()
        batch_data_samples[0].gt_instance_labels = InstanceData(
            keypoint_indices=keypoint_indices,
            keypoint_weights=keypoint_weights).to_tensor()

        feats = [torch.rand(1, 32, 64, 64)]
        head = AssociativeEmbeddingHead(
            in_channels=32,
            num_keypoints=17,
            tag_dim=1,
            tag_per_keypoint=True,
            deconv_out_channels=None)

        losses = head.loss(feats, batch_data_samples)
        for name in ['loss_kpt', 'loss_pull', 'loss_push']:
            self.assertIn(name, losses)
            self.assertIsInstance(losses[name], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
