# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from ..webcam_utils import Message
from .builder import NODES
from .node import Node


@NODES.register_module()
class PoseVisualizer(Node):

    def __init__(self,
                 name: Optional[str] = None,
                 hotkeys=None,
                 enable_key=None,
                 frame_buffer: str = '_frame_',
                 result_buffer: Optional[str] = None,
                 output_buffer: str = '_output_'):
        super().__init__(name=name, hotkeys=hotkeys, enable_key=enable_key)

        # Register buffers
        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        if result_buffer is None:
            self.show_result = False
        else:
            self.register_input_buffer(result_buffer, 'result')
            self.show_result = True
        self.register_output_buffer(output_buffer)

    def _show_result(self, frame_msg, result_msg):
        raise NotImplementedError

    def process(self, input_msgs):
        frame_msg = input_msgs['frame']

        if self.show_result:
            result_msg = input_msgs['result']
            output_msg = self.show_result(frame_msg, result_msg)

        else:
            output_msg = Message(data=frame_msg.data)

        return output_msg
