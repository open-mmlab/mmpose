# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import mmcv
import numpy as np
from mmengine import Config

from mmpose.apis.webcam.nodes import ObjectVisualizerNode
from mmpose.apis.webcam.utils.message import FrameMessage
from mmpose.datasets.datasets.utils import parse_pose_metainfo


class TestObjectVisualizerNode(unittest.TestCase):

    def _get_input_msg(self):

        msg = FrameMessage(None)

        image_path = 'tests/data/coco/000000000785.jpg'
        image = mmcv.imread(image_path)
        h, w = image.shape[:2]
        msg.set_image(image)

        objects = [
            dict(
                label='person',
                class_id=0,
                bbox=np.array([285.1, 44.4, 510.2, 387.7]),
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

    def test_init(self):
        node = ObjectVisualizerNode(
            name='object visualizer',
            input_buffer='frame',
            output_buffer='vis')

        self.assertEqual(len(node._input_buffers), 1)
        self.assertEqual(len(node._output_buffers), 1)
        self.assertEqual(node._input_buffers[0].buffer_name, 'frame')
        self.assertEqual(node._output_buffers[0].buffer_name, 'vis')

    def test_draw(self):
        # draw all objects with bounding box
        node = ObjectVisualizerNode(
            name='object visualizer',
            input_buffer='frame',
            output_buffer='vis')
        input_msg = self._get_input_msg()
        img_h, img_w = input_msg.get_image().shape[:2]
        self.assertEqual(len(input_msg.get_objects()), 1)

        canvas = node.draw(input_msg)
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape[0], img_h)
        self.assertEqual(canvas.shape[1], img_w)

        # draw all objects with keypoints
        node = ObjectVisualizerNode(
            name='object visualizer',
            input_buffer='frame',
            output_buffer='vis',
            must_have_keypoint=True)
        canvas = node.draw(input_msg)
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape[0], img_h)
        self.assertEqual(canvas.shape[1], img_w)


if __name__ == '__main__':
    unittest.main()
