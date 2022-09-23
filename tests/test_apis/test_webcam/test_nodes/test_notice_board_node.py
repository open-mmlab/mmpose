# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import mmcv
import numpy as np

from mmpose.apis.webcam.nodes import NoticeBoardNode
from mmpose.apis.webcam.utils.message import FrameMessage


class TestNoticeBoardNode(unittest.TestCase):

    def _get_input_msg(self):

        msg = FrameMessage(None)

        image_path = 'tests/data/coco/000000000785.jpg'
        image = mmcv.imread(image_path)
        h, w = image.shape[:2]
        msg.set_image(image)

        return msg

    def test_init(self):
        node = NoticeBoardNode(
            name='instruction', input_buffer='vis', output_buffer='vis_notice')

        self.assertEqual(len(node._input_buffers), 1)
        self.assertEqual(len(node._output_buffers), 1)
        self.assertEqual(node._input_buffers[0].buffer_name, 'vis')
        self.assertEqual(node._output_buffers[0].buffer_name, 'vis_notice')
        self.assertEqual(len(node.content_lines), 1)

        node = NoticeBoardNode(
            name='instruction',
            input_buffer='vis',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:',
                '"v": Pose estimation result visualization',
                '"s": Sunglasses effect B-)', '"b": Big-eye effect 0_0',
                '"h": Show help information',
                '"m": Show diagnostic information', '"q": Exit'
            ])
        self.assertEqual(len(node.content_lines), 9)

    def test_draw(self):
        node = NoticeBoardNode(
            name='instruction', input_buffer='vis', output_buffer='vis_notice')
        input_msg = self._get_input_msg()
        img_h, img_w = input_msg.get_image().shape[:2]

        canvas = node.draw(input_msg)
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape[0], img_h)
        self.assertEqual(canvas.shape[1], img_w)


if __name__ == '__main__':
    unittest.main()
