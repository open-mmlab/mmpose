# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest

import mmcv
from mmengine import Config

from mmpose.apis.webcam import WebcamExecutor


class TestWebcamExecutor(unittest.TestCase):

    def _get_config(self):
        config = Config.fromfile('demo/webcam_cfg/test_camera.py').executor_cfg
        config.camera_id = 'tests/data/posetrack18/videos/' \
                           '000001_mpiinew_test/000001_mpiinew_test.mp4'
        return config

    def test_init(self):

        executor = WebcamExecutor(**self._get_config())

        self.assertEqual(len(executor.node_list), 2)
        self.assertEqual(executor.node_list[0].name, 'monitor')
        self.assertEqual(executor.node_list[1].name, 'recorder')

    def test_run(self):
        executor = WebcamExecutor(**self._get_config())

        if os.path.exists('webcam_output.mp4'):
            os.remove('webcam_output.mp4')

        executor.run()

        # check the properties of output video
        self.assertTrue(os.path.exists('webcam_output.mp4'))
        video = mmcv.VideoReader('webcam_output.mp4')
        self.assertGreaterEqual(video.frame_cnt, 1)
        self.assertEqual(video.fps, 30)
        os.remove('webcam_output.mp4')


if __name__ == '__main__':
    unittest.main()
