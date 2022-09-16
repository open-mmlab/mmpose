# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmengine import Config

from mmpose.apis.webcam import WebcamExecutor


class TestWebcamExecutor(unittest.TestCase):

    def setUp(self) -> None:
        config = Config.fromfile('demo/webcam_cfg/test_camera.py').executor_cfg
        config.camera_id = 'tests/data/posetrack18/videos/' \
                           '000001_mpiinew_test/000001_mpiinew_test.mp4'
        self.executor = WebcamExecutor(**config)

    def test_init(self):

        self.assertEqual(len(self.executor.node_list), 2)
        self.assertEqual(self.executor.node_list[0].name, 'monitor')
        self.assertEqual(self.executor.node_list[1].name, 'recorder')


if __name__ == '__main__':
    unittest.main()
