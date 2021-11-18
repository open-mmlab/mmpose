# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from tools.webcam.webcam_utils import Message

from ..builder import NODES
from .node import Node


@NODES.register_module()
class PoseVisualizer(Node):

    def __init__(self,
                 name: Optional[str] = None,
                 hotkeys=None,
                 enable_key=None,
                 frame_buffer: str = '_frame',
                 inference_buffer: Optional[str] = None,
                 output_buffer: str = '_output'):
        super().__init__(name=name, hotkeys=hotkeys, enable_key=enable_key)

        # Register buffers
        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        if inference_buffer:
            self.register_input_buffer(inference_buffer, 'inference')
        self.register_output_buffer(output_buffer)

    def process(self, input_msgs):
        frame_msg = input_msgs['frame']
        inference_msg = input_msgs['inference']

        if inference_msg is not None:
            pass

        output_msg = Message(data=frame_msg.data)

        return output_msg
