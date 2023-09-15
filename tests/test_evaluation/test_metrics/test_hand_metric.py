# Copyright (c) OpenMMLab. All rights reserved.
import json
import tempfile
from unittest import TestCase

import numpy as np
from mmengine.fileio import load
from xtcocotools.coco import COCO

from mmpose.codecs.utils import camera_to_pixel
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.evaluation import InterHandMetric


class TestInterHandMetric(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.tmp_dir = tempfile.TemporaryDirectory()

        self.ann_file = 'tests/data/interhand2.6m/test_interhand2.6m_data.json'
        meta_info = dict(from_file='configs/_base_/datasets/interhand3d.py')
        self.dataset_meta = parse_pose_metainfo(meta_info)
        self.coco = COCO(self.ann_file)

        self.joint_file = ('tests/data/interhand2.6m/'
                           'test_interhand2.6m_joint_3d.json')
        with open(self.joint_file, 'r') as f:
            self.joints = json.load(f)

        self.camera_file = ('tests/data/interhand2.6m/'
                            'test_interhand2.6m_camera.json')
        with open(self.camera_file, 'r') as f:
            self.cameras = json.load(f)

        self.topdown_data = self._convert_ann_to_topdown_batch_data(
            self.ann_file)
        assert len(self.topdown_data) == 4
        self.target = {
            'MPJPE_all': 0.0,
            'MPJPE_interacting': 0.0,
            'MPJPE_single': 0.0,
            'MRRPE': 0.0,
            'HandednessAcc': 1.0
        }

    def encode_handtype(self, hand_type):
        if hand_type == 'right':
            return np.array([[1, 0]], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([[0, 1]], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([[1, 1]], dtype=np.float32)
        else:
            assert 0, f'Not support hand type: {hand_type}'

    def _convert_ann_to_topdown_batch_data(self, ann_file):
        """Convert annotations to topdown-style batch data."""
        topdown_data = []
        db = load(ann_file)
        num_keypoints = 42
        imgid2info = dict()
        for img in db['images']:
            imgid2info[img['id']] = img
        for ann in db['annotations']:
            image_id = ann['image_id']
            img = imgid2info[image_id]
            frame_idx = str(img['frame_idx'])
            capture_id = str(img['capture'])
            camera_name = img['camera']

            camera_pos = np.array(
                self.cameras[capture_id]['campos'][camera_name],
                dtype=np.float32)
            camera_rot = np.array(
                self.cameras[capture_id]['camrot'][camera_name],
                dtype=np.float32)
            focal = np.array(
                self.cameras[capture_id]['focal'][camera_name],
                dtype=np.float32)
            principal_pt = np.array(
                self.cameras[capture_id]['princpt'][camera_name],
                dtype=np.float32)
            joint_world = np.array(
                self.joints[capture_id][frame_idx]['world_coord'],
                dtype=np.float32)
            joint_valid = np.array(
                ann['joint_valid'], dtype=np.float32).flatten()

            keypoints_cam = np.dot(
                camera_rot,
                joint_world.transpose(1, 0) -
                camera_pos.reshape(3, 1)).transpose(1, 0)
            joint_img = camera_to_pixel(
                keypoints_cam,
                focal[0],
                focal[1],
                principal_pt[0],
                principal_pt[1],
                shift=True)[:, :2]

            abs_depth = [keypoints_cam[20, 2], keypoints_cam[41, 2]]

            rel_root_depth = keypoints_cam[41, 2] - keypoints_cam[20, 2]

            joint_valid[:20] *= joint_valid[20]
            joint_valid[21:] *= joint_valid[41]

            joints_3d = np.zeros((num_keypoints, 3),
                                 dtype=np.float32).reshape(1, -1, 3)
            joints_3d[..., :2] = joint_img
            joints_3d[..., :21,
                      2] = keypoints_cam[:21, 2] - keypoints_cam[20, 2]
            joints_3d[..., 21:,
                      2] = keypoints_cam[21:, 2] - keypoints_cam[41, 2]
            joints_3d_visible = np.minimum(1, joint_valid.reshape(-1, 1))
            joints_3d_visible = joints_3d_visible.reshape(1, -1)

            gt_instances = {
                'keypoints_cam': keypoints_cam.reshape(1, -1, 3),
                'keypoints_visible': joints_3d_visible,
            }
            pred_instances = {
                'keypoints': joints_3d,
                'hand_type': self.encode_handtype(ann['hand_type']),
                'rel_root_depth': rel_root_depth,
            }

            data = {'inputs': None}
            data_sample = {
                'id': ann['id'],
                'img_id': ann['image_id'],
                'gt_instances': gt_instances,
                'pred_instances': pred_instances,
                'hand_type': self.encode_handtype(ann['hand_type']),
                'hand_type_valid': np.array([ann['hand_type_valid']]),
                'abs_depth': abs_depth,
                'focal': focal,
                'principal_pt': principal_pt,
            }

            # batch size = 1
            data_batch = [data]
            data_samples = [data_sample]
            topdown_data.append((data_batch, data_samples))

        return topdown_data

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        """test metric init method."""
        # test modes option
        with self.assertRaisesRegex(ValueError, '`mode` should be'):
            _ = InterHandMetric(modes=['invalid'])

    def test_topdown_evaluate(self):
        """test topdown-style COCO metric evaluation."""
        # case 1: modes='MPJPE'
        metric = InterHandMetric(modes=['MPJPE'])
        metric.dataset_meta = self.dataset_meta

        # process samples
        for data_batch, data_samples in self.topdown_data:
            metric.process(data_batch, data_samples)

        eval_results = metric.evaluate(size=len(self.topdown_data))

        for metric, err in eval_results.items():
            self.assertAlmostEqual(err, self.target[metric], places=4)

        # case 2: modes='MRRPE'
        metric = InterHandMetric(modes=['MRRPE'])
        metric.dataset_meta = self.dataset_meta

        # process samples
        for data_batch, data_samples in self.topdown_data:
            metric.process(data_batch, data_samples)

        eval_results = metric.evaluate(size=len(self.topdown_data))

        for metric, err in eval_results.items():
            self.assertAlmostEqual(err, self.target[metric], places=4)

        # case 2: modes='HandednessAcc'
        metric = InterHandMetric(modes=['HandednessAcc'])
        metric.dataset_meta = self.dataset_meta

        # process samples
        for data_batch, data_samples in self.topdown_data:
            metric.process(data_batch, data_samples)

        eval_results = metric.evaluate(size=len(self.topdown_data))

        for metric, err in eval_results.items():
            self.assertAlmostEqual(err, self.target[metric], places=4)
