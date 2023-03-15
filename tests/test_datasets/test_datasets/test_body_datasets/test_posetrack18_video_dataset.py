# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np

from mmpose.datasets.datasets.body import PoseTrack18VideoDataset


class TestPoseTrack18VideoDataset(TestCase):

    def build_posetrack18_video_dataset(self, **kwargs):

        cfg = dict(
            ann_file='annotations/test_posetrack18_val.json',
            bbox_file=None,
            data_mode='topdown',
            frame_weights=[0.0, 1.0],
            frame_sampler_mode='random',
            frame_range=[-2, 2],
            num_sampled_frame=1,
            frame_indices=[-2, -1, 0, 1, 2],
            ph_fill_len=6,
            data_root='tests/data/posetrack18',
            pipeline=[],
            test_mode=False)

        cfg.update(kwargs)
        return PoseTrack18VideoDataset(**cfg)

    def check_data_info_keys(self,
                             data_info: dict,
                             data_mode: str = 'topdown'):
        if data_mode == 'topdown':
            expected_keys = dict(
                img_id=int,
                # mind this difference: img_path is a list
                img_path=list,
                bbox=np.ndarray,
                bbox_score=np.ndarray,
                keypoints=np.ndarray,
                keypoints_visible=np.ndarray,
                # mind this difference
                frame_weights=np.ndarray,
                id=int)
        elif data_mode == 'bottomup':
            expected_keys = dict(
                img_id=int,
                img_path=list,
                bbox=np.ndarray,
                bbox_score=np.ndarray,
                keypoints=np.ndarray,
                keypoints_visible=np.ndarray,
                invalid_segs=list,
                frame_weights=np.ndarray,
                id=list)
        else:
            raise ValueError(f'Invalid data_mode {data_mode}')

        for key, type_ in expected_keys.items():
            self.assertIn(key, data_info)
            self.assertIsInstance(data_info[key], type_, key)

    def check_metainfo_keys(self, metainfo: dict):
        expected_keys = dict(
            dataset_name=str,
            num_keypoints=int,
            keypoint_id2name=dict,
            keypoint_name2id=dict,
            upper_body_ids=list,
            lower_body_ids=list,
            flip_indices=list,
            flip_pairs=list,
            keypoint_colors=np.ndarray,
            num_skeleton_links=int,
            skeleton_links=list,
            skeleton_link_colors=np.ndarray,
            dataset_keypoint_weights=np.ndarray)

        for key, type_ in expected_keys.items():
            self.assertIn(key, metainfo)
            self.assertIsInstance(metainfo[key], type_, key)

    def test_metainfo(self):
        dataset = self.build_posetrack18_video_dataset()
        self.check_metainfo_keys(dataset.metainfo)
        # test dataset_name
        self.assertEqual(dataset.metainfo['dataset_name'], 'posetrack18')

        # test number of keypoints
        num_keypoints = 17
        self.assertEqual(dataset.metainfo['num_keypoints'], num_keypoints)
        self.assertEqual(
            len(dataset.metainfo['keypoint_colors']), num_keypoints)
        self.assertEqual(
            len(dataset.metainfo['dataset_keypoint_weights']), num_keypoints)
        # note that len(sigmas) may be zero if dataset.metainfo['sigmas'] = []
        self.assertEqual(len(dataset.metainfo['sigmas']), num_keypoints)

        # test some extra metainfo
        self.assertEqual(
            len(dataset.metainfo['skeleton_links']),
            len(dataset.metainfo['skeleton_link_colors']))

    def test_topdown(self):
        # test topdown training, frame_sampler_mode = 'random'
        dataset = self.build_posetrack18_video_dataset(
            data_mode='topdown', frame_sampler_mode='random')
        self.assertEqual(len(dataset), 14)
        self.check_data_info_keys(dataset[0])

        # test topdown training, frame_sampler_mode = 'fixed'
        dataset = self.build_posetrack18_video_dataset(
            data_mode='topdown',
            frame_sampler_mode='fixed',
            frame_weights=[0.0, 1.0],
            frame_indices=[-1, 0])
        self.assertEqual(len(dataset), 14)
        self.check_data_info_keys(dataset[0])

        # test topdown testing, frame_sampler_mode = 'random'
        dataset = self.build_posetrack18_video_dataset(
            data_mode='topdown', test_mode=True, frame_sampler_mode='random')
        self.assertEqual(len(dataset), 14)
        self.check_data_info_keys(dataset[0])

        # test topdown testing, frame_sampler_mode = 'fixed'
        dataset = self.build_posetrack18_video_dataset(
            data_mode='topdown',
            test_mode=True,
            frame_sampler_mode='fixed',
            frame_weights=(0.3, 0.1, 0.25, 0.25, 0.1),
            frame_indices=[-2, -1, 0, 1, 2])
        self.assertEqual(len(dataset), 14)
        self.check_data_info_keys(dataset[0])

        # test topdown testing with bbox file, frame_sampler_mode = 'random'
        dataset = self.build_posetrack18_video_dataset(
            test_mode=True,
            frame_sampler_mode='random',
            bbox_file='tests/data/posetrack18/annotations/'
            'test_posetrack18_human_detections.json')
        self.assertEqual(len(dataset), 278)
        self.check_data_info_keys(dataset[0])

        # test topdown testing with bbox file, frame_sampler_mode = 'fixed'
        dataset = self.build_posetrack18_video_dataset(
            test_mode=True,
            frame_sampler_mode='fixed',
            frame_weights=(0.3, 0.1, 0.25, 0.25, 0.1),
            frame_indices=[-2, -1, 0, 1, 2],
            bbox_file='tests/data/posetrack18/annotations/'
            'test_posetrack18_human_detections.json')
        self.assertEqual(len(dataset), 278)
        self.check_data_info_keys(dataset[0])

        # test topdown testing with filter config
        dataset = self.build_posetrack18_video_dataset(
            test_mode=True,
            frame_sampler_mode='fixed',
            frame_weights=(0.3, 0.1, 0.25, 0.25, 0.1),
            frame_indices=[-2, -1, 0, 1, 2],
            bbox_file='tests/data/posetrack18/annotations/'
            'test_posetrack18_human_detections.json',
            filter_cfg=dict(bbox_score_thr=0.3))
        self.assertEqual(len(dataset), 119)

    def test_bottomup(self):
        # test bottomup training
        dataset = self.build_posetrack18_video_dataset(data_mode='bottomup')
        self.assertEqual(len(dataset), 3)
        self.check_data_info_keys(dataset[0], data_mode='bottomup')

        # test bottomup testing
        dataset = self.build_posetrack18_video_dataset(
            data_mode='bottomup',
            test_mode=True,
            frame_sampler_mode='fixed',
            frame_indices=[-2, -1, 0, 1, 2],
            frame_weights=(0.3, 0.1, 0.25, 0.25, 0.1))
        self.assertEqual(len(dataset), 3)
        self.check_data_info_keys(dataset[0], data_mode='bottomup')

    def test_exceptions_and_warnings(self):
        # test invalid frame_weights
        with self.assertRaisesRegex(AssertionError,
                                    'Invalid `frame_weights`:'):
            _ = self.build_posetrack18_video_dataset(frame_weights=[0])

        with self.assertRaisesRegex(AssertionError, 'should sum to 1.0'):
            _ = self.build_posetrack18_video_dataset(frame_weights=[0.2, 0.3])
        with self.assertRaisesRegex(
                AssertionError, 'frame_weight can not be a negative value'):
            _ = self.build_posetrack18_video_dataset(frame_weights=[-0.2, 1.2])

        # test invalid frame_sampler_mode
        with self.assertRaisesRegex(ValueError,
                                    'got invalid frame_sampler_mode'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='invalid')

        # test invalid argument when `frame_sampler_mode = 'random'`
        with self.assertRaisesRegex(AssertionError,
                                    'please specify the `frame_range`'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='random',
                frame_range=None,
            )
        with self.assertRaisesRegex(AssertionError,
                                    'frame_range can not be a negative value'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='random',
                frame_range=-2,
            )
        # correct usage
        _ = self.build_posetrack18_video_dataset(
            frame_sampler_mode='random',
            frame_range=2,
        )
        with self.assertRaisesRegex(AssertionError, 'The length must be 2'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='random',
                frame_range=[3],
            )
        with self.assertRaisesRegex(AssertionError, 'Invalid `frame_range`'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='random',
                frame_range=[3, -3],
            )
        with self.assertRaisesRegex(AssertionError,
                                    'Each element must be int'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='random',
                frame_range=[-2, 5.5],
            )
        with self.assertRaisesRegex(
                TypeError,
                'The type of `frame_range` must be int or Sequence'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='random',
                frame_range=dict(low=-2, high=2),
            )

        # test valid number of frames
        with self.assertRaisesRegex(AssertionError,
                                    'please specify `num_sampled_frame`'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='random',
                num_sampled_frame=None,
            )
        with self.assertRaisesRegex(
                AssertionError,
                'does not match the number of sampled adjacent frames'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='random',
                frame_weights=[0.2, 0.3, 0.5],
                num_sampled_frame=1,
            )

        # test invalid argument when `frame_sampler_mode = 'fixed'`
        with self.assertRaisesRegex(AssertionError,
                                    'please specify the `frame_indices`'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='fixed',
                frame_indices=None,
            )
        with self.assertRaisesRegex(
                AssertionError, 'does not match the length of frame_indices'):
            _ = self.build_posetrack18_video_dataset(
                frame_sampler_mode='fixed',
                frame_weights=[0.5, 0.3, 0.2],
                frame_indices=[-2, -1, 0, 1, 2],
            )

        with self.assertRaisesRegex(ValueError, 'got invalid data_mode'):
            _ = self.build_posetrack18_video_dataset(data_mode='invalid')

        with self.assertRaisesRegex(
                ValueError,
                '"bbox_file" is only supported when `test_mode==True`'):
            _ = self.build_posetrack18_video_dataset(
                test_mode=False,
                bbox_file='tests/data/posetrack18/annotations/'
                'test_posetrack18_human_detections.json')

        with self.assertRaisesRegex(
                ValueError, '"bbox_file" is only supported in topdown mode'):
            _ = self.build_posetrack18_video_dataset(
                data_mode='bottomup',
                test_mode=True,
                bbox_file='tests/data/posetrack18/annotations/'
                'test_posetrack18_human_detections.json')

        with self.assertRaisesRegex(
                ValueError,
                '"bbox_score_thr" is only supported in topdown mode'):
            _ = self.build_posetrack18_video_dataset(
                data_mode='bottomup',
                test_mode=True,
                filter_cfg=dict(bbox_score_thr=0.3))
