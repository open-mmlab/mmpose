# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

import numpy as np
from mmengine.fileio import load

from mmpose.codecs import MotionBERTLabel
from mmpose.registry import KEYPOINT_CODECS


class TestMotionBERTLabel(TestCase):

    def get_camera_param(self, imgname, camera_param) -> dict:
        """Get camera parameters of a frame by its image name."""
        subj, rest = osp.basename(imgname).split('_', 1)
        action, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        return camera_param[(subj, camera)]

    def build_pose_lifting_label(self, **kwargs):
        cfg = dict(type='MotionBERTLabel', num_keypoints=17)
        cfg.update(kwargs)
        return KEYPOINT_CODECS.build(cfg)

    def setUp(self) -> None:
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 2)) * [1000, 1002]
        keypoints = np.round(keypoints).astype(np.float32)
        keypoints_visible = np.random.randint(2, size=(1, 17))
        lifting_target = (0.1 + 0.8 * np.random.rand(1, 17, 3))
        lifting_target_visible = np.random.randint(
            2, size=(
                1,
                17,
            ))
        encoded_wo_sigma = np.random.rand(1, 17, 3)

        camera_param = load('tests/data/h36m/cameras.pkl')
        camera_param = self.get_camera_param(
            'S1/S1_Directions_1.54138969/S1_Directions_1.54138969_000001.jpg',
            camera_param)
        factor = 0.1 + 5 * np.random.rand(1, )

        self.data = dict(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            lifting_target=lifting_target,
            lifting_target_visible=lifting_target_visible,
            camera_param=camera_param,
            factor=factor,
            encoded_wo_sigma=encoded_wo_sigma)

    def test_build(self):
        codec = self.build_pose_lifting_label()
        self.assertIsInstance(codec, MotionBERTLabel)

    def test_encode(self):
        keypoints = self.data['keypoints']
        keypoints_visible = self.data['keypoints_visible']
        lifting_target = self.data['lifting_target']
        lifting_target_visible = self.data['lifting_target_visible']
        camera_param = self.data['camera_param']
        factor = self.data['factor']

        # test default settings
        codec = self.build_pose_lifting_label()
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible, camera_param, factor)

        self.assertEqual(encoded['keypoint_labels'].shape, (1, 17, 2))
        self.assertEqual(encoded['lifting_target_label'].shape, (1, 17, 3))
        self.assertEqual(encoded['lifting_target_weight'].shape, (
            1,
            17,
        ))

        # test concatenating visibility
        codec = self.build_pose_lifting_label(concat_vis=True)
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible, camera_param, factor)

        self.assertEqual(encoded['keypoint_labels'].shape, (1, 17, 3))
        self.assertEqual(encoded['lifting_target_label'].shape, (1, 17, 3))

    def test_decode(self):
        encoded_wo_sigma = self.data['encoded_wo_sigma']
        camera_param = self.data['camera_param']

        # test default settings
        codec = self.build_pose_lifting_label()

        decoded, scores = codec.decode(encoded_wo_sigma)

        self.assertEqual(decoded.shape, (1, 17, 3))
        self.assertEqual(scores.shape, (1, 17))

        # test denormalize according to image shape
        codec = self.build_pose_lifting_label()

        decoded, scores = codec.decode(
            encoded_wo_sigma,
            w=np.array([camera_param['w']]),
            h=np.array([camera_param['h']]))

        self.assertEqual(decoded.shape, (1, 17, 3))
        self.assertEqual(scores.shape, (1, 17))

        # test with factor
        codec = self.build_pose_lifting_label()

        decoded, scores = codec.decode(
            encoded_wo_sigma, factor=np.array([0.23]))

        self.assertEqual(decoded.shape, (1, 17, 3))
        self.assertEqual(scores.shape, (1, 17))

    def test_cicular_verification(self):
        keypoints_visible = self.data['keypoints_visible']
        lifting_target = self.data['lifting_target']
        lifting_target_visible = self.data['lifting_target_visible']
        camera_param = self.data['camera_param']

        # test denormalize according to image shape
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 3))
        codec = self.build_pose_lifting_label()
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible, camera_param)

        _keypoints, _ = codec.decode(
            encoded['keypoint_labels'],
            w=np.array([camera_param['w']]),
            h=np.array([camera_param['h']]))

        keypoints[..., :, :] = keypoints[..., :, :] - keypoints[..., 0, :]

        self.assertTrue(
            np.allclose(keypoints[..., :2] / 1000, _keypoints[..., :2]))

        # test with factor
        keypoints = (0.1 + 0.8 * np.random.rand(1, 17, 3))
        codec = self.build_pose_lifting_label()
        encoded = codec.encode(keypoints, keypoints_visible, lifting_target,
                               lifting_target_visible, camera_param)

        _keypoints, _ = codec.decode(
            encoded['keypoint_labels'],
            w=np.array([camera_param['w']]),
            h=np.array([camera_param['h']]),
            factor=encoded['factor'])

        keypoints *= encoded['factor']
        keypoints[..., :, :] = keypoints[..., :, :] - keypoints[..., 0, :]

        self.assertTrue(
            np.allclose(keypoints[..., :2] / 1000, _keypoints[..., :2]))
