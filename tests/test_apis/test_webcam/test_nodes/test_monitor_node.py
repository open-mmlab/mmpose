# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import mmcv

from mmpose.apis.webcam.nodes import MonitorNode
from mmpose.apis.webcam.utils.message import FrameMessage


class TestMonitorNode(unittest.TestCase):

    def _get_input_msg(self):

        msg = FrameMessage(None)

        image_path = 'tests/data/coco/000000000785.jpg'
        image = mmcv.imread(image_path)
        msg.set_image(image)

        objects = [dict(label='human')]
        msg.update_objects(objects)

        return msg

    def test_init(self):
        node = MonitorNode(
            name='monitor', input_buffer='_frame_', output_buffer='display')
        self.assertEqual(len(node._input_buffers), 1)
        self.assertEqual(len(node._output_buffers), 1)
        self.assertEqual(node._input_buffers[0].buffer_name, '_frame_')
        self.assertEqual(node._output_buffers[0].buffer_name, 'display')

        # test initialization with given ignore_items
        node = MonitorNode(
            name='monitor',
            input_buffer='_frame_',
            output_buffer='display',
            ignore_items=['ignore_item'])
        self.assertEqual(len(node.ignore_items), 1)
        self.assertEqual(node.ignore_items[0], 'ignore_item')

    def test_process(self):
        node = MonitorNode(
            name='monitor', input_buffer='_frame_', output_buffer='display')

        input_msg = self._get_input_msg()
        self.assertEqual(len(input_msg.get_route_info()), 0)
        img_shape = input_msg.get_image().shape

        output_msg = node.process(dict(input=input_msg))
        # 'System Info' will be added into route_info
        self.assertEqual(len(output_msg.get_route_info()), 1)
        self.assertEqual(output_msg.get_image().shape, img_shape)

    def test_bypass(self):
        node = MonitorNode(
            name='monitor', input_buffer='_frame_', output_buffer='display')
        input_msg = self._get_input_msg()
        self.assertEqual(len(input_msg.get_route_info()), 0)

        output_msg = node.bypass(dict(input=input_msg))
        # output_msg should be identity with input_msg
        self.assertEqual(len(output_msg.get_route_info()), 0)


if __name__ == '__main__':
    unittest.main()
