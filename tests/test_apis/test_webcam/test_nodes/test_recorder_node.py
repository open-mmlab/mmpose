# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest

import mmcv

from mmpose.apis.webcam.nodes import RecorderNode
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
        node = RecorderNode(
            name='recorder',
            out_video_file='webcam_output.mp4',
            input_buffer='display',
            output_buffer='_display_')
        self.assertEqual(len(node._input_buffers), 1)
        self.assertEqual(len(node._output_buffers), 1)
        self.assertEqual(node._input_buffers[0].buffer_name, 'display')
        self.assertEqual(node._output_buffers[0].buffer_name, '_display_')
        self.assertTrue(node.t_record.is_alive())

    def test_process(self):
        node = RecorderNode(
            name='recorder',
            out_video_file='webcam_output.mp4',
            input_buffer='display',
            output_buffer='_display_',
            buffer_size=1)

        if os.path.exists('webcam_output.mp4'):
            os.remove('webcam_output.mp4')

        input_msg = self._get_input_msg()
        node.process(dict(input=input_msg))
        self.assertEqual(node.queue.qsize(), 1)

        # process 5 frames in total.
        # the first frame has been processed above
        for _ in range(4):
            node.process(dict(input=input_msg))
        node.on_exit()

        # check the properties of output video
        self.assertTrue(os.path.exists('webcam_output.mp4'))
        video = mmcv.VideoReader('webcam_output.mp4')
        self.assertEqual(video.frame_cnt, 5)
        self.assertEqual(video.fps, 30)
        video.vcap.release()
        os.remove('webcam_output.mp4')


if __name__ == '__main__':
    unittest.main()
