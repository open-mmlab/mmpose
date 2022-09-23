# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import mmcv
import numpy as np

from mmpose.apis.webcam.nodes import MonitorNode
from mmpose.apis.webcam.utils.message import FrameMessage, Message


class TestMessage(unittest.TestCase):

    def _get_monitor_node(self):
        return MonitorNode(
            name='monitor', input_buffer='_frame_', output_buffer='display')

    def _get_image(self):
        image_path = 'tests/data/coco/000000000785.jpg'
        image = mmcv.imread(image_path)
        return image

    def test_message(self):
        msg = Message()

        with self.assertWarnsRegex(
                Warning, '`node_name` and `node_type` will be '
                'overridden if node is provided.'):
            node = self._get_monitor_node()
            msg.update_route_info(node=node, node_name='monitor')

        route_info = msg.get_route_info()
        self.assertEqual(len(route_info), 1)
        self.assertEqual(route_info[0]['node'], 'monitor')

        msg.set_route_info([dict(node='recorder', node_type='RecorderNode')])
        msg.merge_route_info(route_info)
        route_info = msg.get_route_info()
        self.assertEqual(len(route_info), 2)
        self.assertEqual(route_info[1]['node'], 'monitor')

    def test_frame_message(self):
        msg = FrameMessage(None)

        # test set/get image
        self.assertIsInstance(msg.data, dict)
        self.assertIsNone(msg.get_image())

        msg.set_image(self._get_image())
        self.assertIsInstance(msg.get_image(), np.ndarray)

        # test set/get objects
        objects = msg.get_objects()
        self.assertEqual(len(objects), 0)

        objects = [dict(label='cat'), dict(label='dog')]
        msg.update_objects(objects)
        dog_objects = msg.get_objects(lambda x: x['label'] == 'dog')
        self.assertEqual(len(dog_objects), 1)

        msg.set_objects(objects[:1])
        dog_objects = msg.get_objects(lambda x: x['label'] == 'dog')
        self.assertEqual(len(dog_objects), 0)


if __name__ == '__main__':
    unittest.main()
