# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.codecs import ImagePoseLifting
from mmpose.registry import KEYPOINT_CODECS


class TestImagePoseLifting(TestCase):

    def setUp(self) -> None:
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 2)) * [192, 256]
        keypoints = np.round(keypoints).astype(np.float32)
        keypoints_visible = np.random.randint(2, size=(1, 17))
        lifting_target = (0.1 + 0.8 * np.random.rand(17, 3))
        lifting_target_visible = np.random.randint(2, size=(17, ))
        encoded_wo_sigma = np.random.rand(1, 17, 3)

        self.keypoints_mean = np.random.rand(17, 2).astype(np.float32)
        self.keypoints_std = np.random.rand(17, 2).astype(np.float32) + 1e-6
        self.target_mean = np.random.rand(17, 3).astype(np.float32)
        self.target_std = np.random.rand(17, 3).astype(np.float32) + 1e-6

        self.data = dict(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            lifting_target=lifting_target,
            lifting_target_visible=lifting_target_visible,
            encoded_wo_sigma=encoded_wo_sigma)

    def build_pose_lifting_label(self, **kwargs):
        cfg = dict(type='ImagePoseLifting', num_keypoints=17, root_index=0)
        cfg.update(kwargs)
        return KEYPOINT_CODECS.build(cfg)

    def test_build(self):
        codec = self.build_pose_lifting_label()
        self.assertIsInstance(codec, ImagePoseLifting)

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']
        lifting_target = self.data['lifting_target']
        lifting_target_visible = self.data['lifting_target_visible']

        # test default settings
        codec = self.build_pose_lifting_label()
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible)

        self.assertEqual(encoded['keypoint_labels'].shape, (1, 17, 2))
        self.assertEqual(encoded['lifting_target_label'].shape, (17, 3))
        self.assertEqual(encoded['lifting_target_weights'].shape, (17, ))
        self.assertEqual(encoded['trajectory_weights'].shape, (17, ))
        self.assertEqual(encoded['target_root'].shape, (3, ))

        # test removing root
        codec = self.build_pose_lifting_label(
            remove_root=True, save_index=True)
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible)

        self.assertTrue('target_root_removed' in encoded
                        and 'target_root_index' in encoded)
        self.assertEqual(encoded['lifting_target_weights'].shape, (16, ))
        self.assertEqual(encoded['keypoint_labels'].shape, (1, 17, 2))
        self.assertEqual(encoded['lifting_target_label'].shape, (16, 3))
        self.assertEqual(encoded['target_root'].shape, (3, ))

        # test normalization
        codec = self.build_pose_lifting_label(
            keypoints_mean=self.keypoints_mean,
            keypoints_std=self.keypoints_std,
            target_mean=self.target_mean,
            target_std=self.target_std)
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible)

        self.assertEqual(encoded['keypoint_labels'].shape, (1, 17, 2))
        self.assertEqual(encoded['lifting_target_label'].shape, (17, 3))

    def test_decode(self):
        lifting_target = self.data['lifting_target']
        encoded_wo_sigma = self.data['encoded_wo_sigma']

        codec = self.build_pose_lifting_label()

        decoded, scores = codec.decode(
            encoded_wo_sigma, target_root=lifting_target[..., 0, :])

        self.assertEqual(decoded.shape, (1, 17, 3))
        self.assertEqual(scores.shape, (1, 17))

        codec = self.build_pose_lifting_label(remove_root=True)

        decoded, scores = codec.decode(
            encoded_wo_sigma, target_root=lifting_target[..., 0, :])

        self.assertEqual(decoded.shape, (1, 18, 3))
        self.assertEqual(scores.shape, (1, 18))

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']
        lifting_target = self.data['lifting_target']
        lifting_target_visible = self.data['lifting_target_visible']

        # test default settings
        codec = self.build_pose_lifting_label()
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible)

        _keypoints, _ = codec.decode(
            np.expand_dims(encoded['lifting_target_label'], axis=0),
            target_root=lifting_target[..., 0, :])

        self.assertTrue(
            np.allclose(
                np.expand_dims(lifting_target, axis=0), _keypoints, atol=5.))

        # test removing root
        codec = self.build_pose_lifting_label(remove_root=True)
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible)

        _keypoints, _ = codec.decode(
            np.expand_dims(encoded['lifting_target_label'], axis=0),
            target_root=lifting_target[..., 0, :])

        self.assertTrue(
            np.allclose(
                np.expand_dims(lifting_target, axis=0), _keypoints, atol=5.))

        # test normalization
        codec = self.build_pose_lifting_label(
            keypoints_mean=self.keypoints_mean,
            keypoints_std=self.keypoints_std,
            target_mean=self.target_mean,
            target_std=self.target_std)
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible)

        _keypoints, _ = codec.decode(
            np.expand_dims(encoded['lifting_target_label'], axis=0),
            target_root=lifting_target[..., 0, :])

        self.assertTrue(
            np.allclose(
                np.expand_dims(lifting_target, axis=0), _keypoints, atol=5.))
