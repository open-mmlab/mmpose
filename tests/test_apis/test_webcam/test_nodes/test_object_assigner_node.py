# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest

import mmcv
import numpy as np

from mmpose.apis.webcam.nodes import ObjectAssignerNode
from mmpose.apis.webcam.utils.message import FrameMessage


class TestObjectAssignerNode(unittest.TestCase):

    def _get_input_msg(self, with_object: bool = False):

        msg = FrameMessage(None)

        image_path = 'tests/data/coco/000000000785.jpg'
        image = mmcv.imread(image_path)
        msg.set_image(image)

        if with_object:
            objects = [
                dict(
                    label='person',
                    class_id=0,
                    bbox=np.array([285.1, 44.4, 510.2, 387.7]))
            ]
            msg.update_objects(objects)

        return msg

    def test_init(self):
        node = ObjectAssignerNode(
            name='object assigner',
            frame_buffer='_frame_',
            object_buffer='pred_result',
            output_buffer='frame')

        self.assertEqual(len(node._input_buffers), 2)
        self.assertEqual(len(node._output_buffers), 1)
        self.assertEqual(node._input_buffers[0].buffer_name, 'pred_result')
        self.assertEqual(node._input_buffers[1].buffer_name, '_frame_')
        self.assertEqual(node._output_buffers[0].buffer_name, 'frame')

    def test_process(self):
        node = ObjectAssignerNode(
            name='object assigner',
            frame_buffer='_frame_',
            object_buffer='pred_result',
            output_buffer='frame')

        frame_msg = self._get_input_msg()
        object_msg = self._get_input_msg(with_object=True)
        self.assertEqual(len(frame_msg.get_objects()), 0)
        self.assertEqual(len(object_msg.get_objects()), 1)

        # node.synchronous is False
        output_msg = node.process(dict(frame=frame_msg, object=object_msg))
        objects = output_msg.get_objects()
        self.assertEqual(id(frame_msg), id(output_msg))
        self.assertEqual(objects[0]['_id_'],
                         object_msg.get_objects()[0]['_id_'])

        # object_message is None
        # take a pause to increase the interval of messages' timestamp
        # to avoid ZeroDivisionError when computing fps in `process`
        time.sleep(1 / 30.0)
        frame_msg = self._get_input_msg()
        output_msg = node.process(dict(frame=frame_msg, object=None))
        objects = output_msg.get_objects()
        self.assertEqual(objects[0]['_id_'],
                         object_msg.get_objects()[0]['_id_'])

        # node.synchronous is True
        node.synchronous = True
        time.sleep(1 / 30.0)
        frame_msg = self._get_input_msg()
        object_msg = self._get_input_msg(with_object=True)
        output_msg = node.process(dict(frame=frame_msg, object=object_msg))
        self.assertEqual(len(frame_msg.get_objects()), 0)
        self.assertEqual(id(object_msg), id(output_msg))


if __name__ == '__main__':
    unittest.main()
