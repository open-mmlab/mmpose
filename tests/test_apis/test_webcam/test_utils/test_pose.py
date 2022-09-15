# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmengine import Config

from mmpose.apis.webcam.utils.pose import (get_eye_keypoint_ids,
                                           get_face_keypoint_ids,
                                           get_hand_keypoint_ids,
                                           get_mouth_keypoint_ids,
                                           get_wrist_keypoint_ids)
from mmpose.datasets.datasets.utils import parse_pose_metainfo


class TestGetKeypointIds(unittest.TestCase):

    def setUp(self) -> None:
        datasets_meta = dict(
            coco=Config.fromfile('configs/_base_/datasets/coco.py'),
            coco_wholebody=Config.fromfile(
                'configs/_base_/datasets/coco_wholebody.py'),
            animalpose=Config.fromfile(
                'configs/_base_/datasets/animalpose.py'),
            ap10k=Config.fromfile('configs/_base_/datasets/ap10k.py'),
            wflw=Config.fromfile('configs/_base_/datasets/wflw.py'),
        )
        self.datasets_meta = {
            key: parse_pose_metainfo(value['dataset_info'])
            for key, value in datasets_meta.items()
        }

    def test_get_eye_keypoint_ids(self):

        # coco dataset
        coco_dataset_meta = self.datasets_meta['coco'].copy()
        left_eye_idx, right_eye_idx = get_eye_keypoint_ids(coco_dataset_meta)
        self.assertEqual(left_eye_idx, 1)
        self.assertEqual(right_eye_idx, 2)

        del coco_dataset_meta['keypoint_name2id']['left_eye']
        left_eye_idx, right_eye_idx = get_eye_keypoint_ids(coco_dataset_meta)
        self.assertEqual(left_eye_idx, 1)
        self.assertEqual(right_eye_idx, 2)

        # animalpose dataset
        animalpose_dataset_meta = self.datasets_meta['animalpose'].copy()
        left_eye_idx, right_eye_idx = get_eye_keypoint_ids(
            animalpose_dataset_meta)
        self.assertEqual(left_eye_idx, 0)
        self.assertEqual(right_eye_idx, 1)

        # dataset without keys `'left_eye'` or `'right_eye'`
        wflw_dataset_meta = self.datasets_meta['wflw'].copy()
        with self.assertRaises(ValueError):
            _ = get_eye_keypoint_ids(wflw_dataset_meta)

    def test_get_face_keypoint_ids(self):

        # coco_wholebody dataset
        wholebody_dataset_meta = self.datasets_meta['coco_wholebody'].copy()
        face_indices = get_face_keypoint_ids(wholebody_dataset_meta)
        for i, ind in enumerate(range(23, 91)):
            self.assertEqual(face_indices[i], ind)

        del wholebody_dataset_meta['keypoint_name2id']['face-0']
        face_indices = get_face_keypoint_ids(wholebody_dataset_meta)
        for i, ind in enumerate(range(23, 91)):
            self.assertEqual(face_indices[i], ind)

        # dataset without keys `'face-x'`
        wflw_dataset_meta = self.datasets_meta['wflw'].copy()
        with self.assertRaises(ValueError):
            _ = get_face_keypoint_ids(wflw_dataset_meta)

    def test_get_wrist_keypoint_ids(self):

        # coco dataset
        coco_dataset_meta = self.datasets_meta['coco'].copy()
        left_wrist_idx, right_wrist_idx = get_wrist_keypoint_ids(
            coco_dataset_meta)
        self.assertEqual(left_wrist_idx, 9)
        self.assertEqual(right_wrist_idx, 10)

        del coco_dataset_meta['keypoint_name2id']['left_wrist']
        left_wrist_idx, right_wrist_idx = get_wrist_keypoint_ids(
            coco_dataset_meta)
        self.assertEqual(left_wrist_idx, 9)
        self.assertEqual(right_wrist_idx, 10)

        # animalpose dataset
        animalpose_dataset_meta = self.datasets_meta['animalpose'].copy()
        left_wrist_idx, right_wrist_idx = get_wrist_keypoint_ids(
            animalpose_dataset_meta)
        self.assertEqual(left_wrist_idx, 16)
        self.assertEqual(right_wrist_idx, 17)

        # ap10k
        ap10k_dataset_meta = self.datasets_meta['ap10k'].copy()
        left_wrist_idx, right_wrist_idx = get_wrist_keypoint_ids(
            ap10k_dataset_meta)
        self.assertEqual(left_wrist_idx, 7)
        self.assertEqual(right_wrist_idx, 10)

        # dataset without keys `'left_wrist'` or `'right_wrist'`
        wflw_dataset_meta = self.datasets_meta['wflw'].copy()
        with self.assertRaises(ValueError):
            _ = get_wrist_keypoint_ids(wflw_dataset_meta)

    def test_get_mouth_keypoint_ids(self):

        # coco_wholebody dataset
        wholebody_dataset_meta = self.datasets_meta['coco_wholebody'].copy()
        mouth_index = get_mouth_keypoint_ids(wholebody_dataset_meta)
        self.assertEqual(mouth_index, 85)

        del wholebody_dataset_meta['keypoint_name2id']['face-62']
        mouth_index = get_mouth_keypoint_ids(wholebody_dataset_meta)
        self.assertEqual(mouth_index, 85)

        # dataset without keys `'face-62'`
        wflw_dataset_meta = self.datasets_meta['wflw'].copy()
        with self.assertRaises(ValueError):
            _ = get_mouth_keypoint_ids(wflw_dataset_meta)

    def test_get_hand_keypoint_ids(self):

        # coco_wholebody dataset
        wholebody_dataset_meta = self.datasets_meta['coco_wholebody'].copy()
        hand_indices = get_hand_keypoint_ids(wholebody_dataset_meta)
        for i, ind in enumerate(range(91, 133)):
            self.assertEqual(hand_indices[i], ind)

        del wholebody_dataset_meta['keypoint_name2id']['left_hand_root']
        hand_indices = get_hand_keypoint_ids(wholebody_dataset_meta)
        for i, ind in enumerate(range(91, 133)):
            self.assertEqual(hand_indices[i], ind)

        # dataset without hand keys
        wflw_dataset_meta = self.datasets_meta['wflw'].copy()
        with self.assertRaises(ValueError):
            _ = get_hand_keypoint_ids(wflw_dataset_meta)


if __name__ == '__main__':
    unittest.main()
