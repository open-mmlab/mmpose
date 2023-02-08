# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import SPR
from mmpose.registry import KEYPOINT_CODECS
from mmpose.testing import get_coco_sample
from mmpose.utils.tensor_utils import to_numpy, to_tensor


class TestSPR(TestCase):

    def setUp(self) -> None:
        pass

    def _make_multi_instance_data(self, data):
        keypoints = data['keypoints']
        keypoints_visible = data['keypoints_visible']

        keypoints_visible[..., 0] = 0

        keypoints_outside = keypoints - keypoints.max(axis=-1, keepdims=True)
        keypoints_outside_visible = np.zeros(keypoints_visible.shape)

        keypoint_overlap = keypoints.mean(
            axis=-1, keepdims=True) + 0.8 * (
                keypoints - keypoints.mean(axis=-1, keepdims=True))
        keypoint_overlap_visible = keypoints_visible

        data['keypoints'] = np.concatenate(
            (keypoints, keypoints_outside, keypoint_overlap), axis=0)
        data['keypoints_visible'] = np.concatenate(
            (keypoints_visible, keypoints_outside_visible,
             keypoint_overlap_visible),
            axis=0)

        return data

    def test_build(self):
        cfg = dict(
            type='SPR',
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=4,
        )
        codec = KEYPOINT_CODECS.build(cfg)
        self.assertIsInstance(codec, SPR)

    def test_encode(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=1)
        data = self._make_multi_instance_data(data)

        # w/o keypoint heatmaps
        codec = SPR(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=4,
        )

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])

        heatmaps = encoded['heatmaps']
        displacements = encoded['displacements']
        heatmap_weights = encoded['heatmap_weights']
        displacement_weights = encoded['displacement_weights']

        self.assertEqual(heatmaps.shape, (1, 128, 128))
        self.assertEqual(heatmap_weights.shape, (1, 128, 128))
        self.assertEqual(displacements.shape, (34, 128, 128))
        self.assertEqual(displacement_weights.shape, (34, 128, 128))

        # w/ keypoint heatmaps
        with self.assertRaises(AssertionError):
            codec = SPR(
                input_size=(512, 512),
                heatmap_size=(128, 128),
                sigma=4,
                generate_keypoint_heatmaps=True,
            )

        codec = SPR(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=(4, 2),
            generate_keypoint_heatmaps=True,
        )

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])

        heatmaps = encoded['heatmaps']
        displacements = encoded['displacements']
        heatmap_weights = encoded['heatmap_weights']
        displacement_weights = encoded['displacement_weights']

        self.assertEqual(heatmaps.shape, (18, 128, 128))
        self.assertEqual(heatmap_weights.shape, (18, 128, 128))
        self.assertEqual(displacements.shape, (34, 128, 128))
        self.assertEqual(displacement_weights.shape, (34, 128, 128))

        # root_type
        with self.assertRaises(ValueError):
            codec = SPR(
                input_size=(512, 512),
                heatmap_size=(128, 128),
                sigma=(4, ),
                root_type='box_center',
            )
            encoded = codec.encode(data['keypoints'],
                                   data['keypoints_visible'])

        codec = SPR(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=(4, ),
            root_type='bbox_center',
        )

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])

        heatmaps = encoded['heatmaps']
        displacements = encoded['displacements']
        heatmap_weights = encoded['heatmap_weights']
        displacement_weights = encoded['displacement_weights']

        self.assertEqual(heatmaps.shape, (1, 128, 128))
        self.assertEqual(heatmap_weights.shape, (1, 128, 128))
        self.assertEqual(displacements.shape, (34, 128, 128))
        self.assertEqual(displacement_weights.shape, (34, 128, 128))

    def test_decode(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=1)

        # decode w/o keypoint heatmaps
        codec = SPR(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=(4, ),
            generate_keypoint_heatmaps=False,
        )

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])
        decoded = codec.decode(
            to_tensor(encoded['heatmaps']),
            to_tensor(encoded['displacements']))

        keypoints, (root_scores, keypoint_scores) = decoded
        self.assertIsNone(keypoint_scores)
        self.assertEqual(keypoints.shape, data['keypoints'].shape)
        self.assertEqual(root_scores.shape, data['keypoints'].shape[:1])

        # decode w/ keypoint heatmaps
        codec = SPR(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=(4, 2),
            generate_keypoint_heatmaps=True,
        )

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])
        decoded = codec.decode(
            to_tensor(encoded['heatmaps']),
            to_tensor(encoded['displacements']))

        keypoints, (root_scores, keypoint_scores) = decoded
        self.assertIsNotNone(keypoint_scores)
        self.assertEqual(keypoints.shape, data['keypoints'].shape)
        self.assertEqual(root_scores.shape, data['keypoints'].shape[:1])
        self.assertEqual(keypoint_scores.shape, data['keypoints'].shape[:2])

    def test_cicular_verification(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=1)

        codec = SPR(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=(4, ),
            generate_keypoint_heatmaps=False,
        )

        encoded = codec.encode(data['keypoints'], data['keypoints_visible'])
        decoded = codec.decode(
            to_tensor(encoded['heatmaps']),
            to_tensor(encoded['displacements']))

        keypoints, _ = decoded
        self.assertTrue(
            np.allclose(to_numpy(keypoints), data['keypoints'], atol=5.))
