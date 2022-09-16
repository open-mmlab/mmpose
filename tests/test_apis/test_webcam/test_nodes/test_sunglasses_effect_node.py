# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import mmcv
import numpy as np
from mmengine import Config

from mmpose.apis.webcam.nodes import SunglassesEffectNode
from mmpose.apis.webcam.utils.message import FrameMessage
from mmpose.datasets.datasets.utils import parse_pose_metainfo


class TestSunglassesEffectNode(unittest.TestCase):

    def setUp(self) -> None:
        self.node = SunglassesEffectNode(
            name='sunglasses',
            input_buffer='vis',
            output_buffer='vis_sunglasses')

    def _get_input_msg(self):

        msg = FrameMessage(None)

        image_path = 'tests/data/coco/000000000785.jpg'
        image = mmcv.imread(image_path)
        h, w = image.shape[:2]
        msg.set_image(image)

        objects = [
            dict(
                keypoints=np.stack((np.random.rand(17) *
                                    (w - 1), np.random.rand(17) * (h - 1)),
                                   axis=1),
                keypoint_scores=np.ones(17),
                dataset_meta=parse_pose_metainfo(
                    Config.fromfile('configs/_base_/datasets/coco.py')
                    ['dataset_info']))
        ]
        msg.update_objects(objects)

        return msg

    def test_process(self):
        input_msg = self._get_input_msg()
        img_h, img_w = input_msg.get_image().shape[:2]
        self.assertEqual(len(input_msg.get_objects()), 1)

        output_msg = self.node.process(dict(input=input_msg))
        canvas = output_msg.get_image()
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape[0], img_h)
        self.assertEqual(canvas.shape[1], img_w)

    def test_bypass(self):
        input_msg = self._get_input_msg()
        img = input_msg.get_image().copy()
        output_msg = self.node.bypass(dict(input=input_msg))
        self.assertTrue((img == output_msg.get_image()).all())


if __name__ == '__main__':
    unittest.main()
