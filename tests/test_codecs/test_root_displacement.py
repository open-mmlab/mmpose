# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmpose.codecs import RootDisplacement
from mmpose.registry import KEYPOINT_CODECS
from mmpose.testing import get_coco_sample


class TestRootDisplacement(TestCase):

    def setUp(self) -> None:
        pass

    def test_build(self):
        cfg = dict(
            type='RootDisplacement',
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=4,
            use_udp=False,
        )
        codec = KEYPOINT_CODECS.build(cfg)
        self.assertIsInstance(codec, RootDisplacement)

    def test_encode(self):
        data = get_coco_sample(img_shape=(512, 512), num_instances=1)

        # w/o keypoint heatmaps
        codec = RootDisplacement(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=4,
            use_udp=False,
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
            codec = RootDisplacement(
                input_size=(512, 512),
                heatmap_size=(128, 128),
                sigma=4,
                generate_keypoint_heatmaps=True,
                use_udp=False,
            )

        codec = RootDisplacement(
            input_size=(512, 512),
            heatmap_size=(128, 128),
            sigma=(4, 2),
            generate_keypoint_heatmaps=True,
            use_udp=False,
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
