# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

import numpy as np
from mmengine.fileio import load

from mmpose.codecs import PoseLiftingLabel
from mmpose.registry import KEYPOINT_CODECS


class TestPoseLiftingLabel(TestCase):

    def get_camera_param(self, imgname, camera_param) -> dict:
        """Get camera parameters of a frame by its image name."""
        subj, rest = osp.basename(imgname).split('_', 1)
        action, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        return camera_param[(subj, camera)]

    def build_pose_lifting_label(self, **kwargs):
        cfg = dict(type='PoseLiftingLabel', num_keypoints=17, root_index=0)
        cfg.update(kwargs)
        return KEYPOINT_CODECS.build(cfg)

    def setUp(self) -> None:
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 2)) * [192, 256]
        keypoints = np.round(keypoints).astype(np.float32)
        keypoints_visible = np.random.randint(2, size=(1, 17))
        target = (0.1 + 0.8 * np.random.rand(17, 3))
        target_visible = np.random.randint(2, size=(17, ))
        encoded_wo_sigma = np.random.rand(1, 17, 3)

        camera_param = load('tests/data/h36m/cameras.pkl')
        camera_param = self.get_camera_param(
            'S1/S1_Directions_1.54138969/S1_Directions_1.54138969_000001.jpg',
            camera_param)

        self.mean = np.random.rand(17, 3).astype(np.float32)
        self.std = np.random.rand(17, 3).astype(np.float32) + 1e-6

        self.data = dict(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            target=target,
            target_visible=target_visible,
            camera_param=camera_param,
            encoded_wo_sigma=encoded_wo_sigma)

    def test_build(self):
        codec = self.build_pose_lifting_label()
        self.assertIsInstance(codec, PoseLiftingLabel)

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']
        target = self.data['target']
        target_visible = self.data['target_visible']
        camera_param = self.data['camera_param']

        # test default settings
        codec = self.build_pose_lifting_label()
        encoded = codec.encode(keypoints, keypoints_visible, target,
                               target_visible, camera_param)

        self.assertEqual(encoded['keypoint_labels'].shape, (17 * 2, 1))
        self.assertEqual(encoded['target_label'].shape, (17, 3))
        self.assertEqual(encoded['target_weights'].shape, (17, ))
        self.assertEqual(encoded['trajectory_weights'].shape, (17, ))
        self.assertEqual(encoded['target_root'].shape, (3, ))

        # test removing root
        codec = self.build_pose_lifting_label(
            remove_root=True, save_index=True)
        encoded = codec.encode(keypoints, keypoints_visible, target,
                               target_visible, camera_param)

        self.assertTrue('target_root_removed' in encoded
                        and 'target_root_index' in encoded)
        self.assertEqual(encoded['target_weights'].shape, (16, ))
        self.assertEqual(encoded['keypoint_labels'].shape, (17 * 2, 1))
        self.assertEqual(encoded['target_label'].shape, (16, 3))
        self.assertEqual(encoded['target_root'].shape, (3, ))

        # test normalizing camera
        codec = self.build_pose_lifting_label(normalize_camera=True)
        encoded = codec.encode(keypoints, keypoints_visible, target,
                               target_visible, camera_param)

        self.assertTrue('camera_param' in encoded)
        scale = np.array(0.5 * camera_param['w'], dtype=np.float32)
        self.assertTrue(
            np.allclose(
                camera_param['f'] / scale,
                encoded['camera_param']['f'],
                atol=4.))

    def test_decode(self):
        target = self.data['target']
        encoded_wo_sigma = self.data['encoded_wo_sigma']

        codec = self.build_pose_lifting_label()

        decoded, scores = codec.decode(
            encoded_wo_sigma,
            restore_global_position=True,
            target_root=target[..., 0, :])

        self.assertEqual(decoded.shape, (1, 17, 3))
        self.assertEqual(scores.shape, (1, 17))

        codec = self.build_pose_lifting_label()

        decoded, scores = codec.decode(
            encoded_wo_sigma,
            restore_global_position=True,
            target_root=target[..., 0, :],
            target_mean=self.mean,
            target_std=self.std)

        self.assertEqual(decoded.shape, (1, 17, 3))
        self.assertEqual(scores.shape, (1, 17))

    def test_cicular_verification(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']
        target = self.data['target']
        target_visible = self.data['target_visible']
        camera_param = self.data['camera_param']

        # test default settings
        codec = self.build_pose_lifting_label()
        encoded = codec.encode(keypoints, keypoints_visible, target,
                               target_visible, camera_param)

        _keypoints, _ = codec.decode(
            np.expand_dims(encoded['target_label'], axis=0),
            restore_global_position=True,
            target_root=target[..., 0, :])

        self.assertTrue(
            np.allclose(np.expand_dims(target, axis=0), _keypoints, atol=5.))

        # test normalization
        codec = self.build_pose_lifting_label()
        encoded = codec.encode(keypoints, keypoints_visible, target,
                               target_visible, camera_param)

        target_label = (encoded['target_label'] - self.mean) / self.std

        _keypoints, _ = codec.decode(
            np.expand_dims(target_label, axis=0),
            restore_global_position=True,
            target_root=target[..., 0, :],
            target_mean=self.mean,
            target_std=self.std)

        self.assertTrue(
            np.allclose(np.expand_dims(target, axis=0), _keypoints, atol=5.))
