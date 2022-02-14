# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np

from ..utils import FrameMessage, Message
from .node import Node


class FrameDrawingNode(Node):
    """Base class for Node that draw on single frame images.

    Args:
        name (str, optional): The node name (also thread name).
        frame_buffer (str): The name of the input buffer.
        output_buffer (str | list): The name(s) of the output buffer(s).
        enable_key (str | int, optional): Set a hot-key to toggle
            enable/disable of the node. If an int value is given, it will be
            treated as an ascii code of a key. Please note:
                1. If enable_key is set, the bypass method need to be
                    overridden to define the node behavior when disabled
                2. Some hot-key has been use for particular use. For example:
                    'q', 'Q' and 27 are used for quit
            Default: None
        enable (bool): Default enable/disable status. Default: True.
    """

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True):

        super().__init__(name=name, enable_key=enable_key)

        # Register buffers
        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        self.register_output_buffer(output_buffer)

        self._enabled = enable

    def process(self, input_msgs: Dict[str, Message]) -> Union[Message, None]:
        frame_msg = input_msgs['frame']

        img = self.draw(frame_msg)
        frame_msg.set_image(img)

        return frame_msg

    def bypass(self, input_msgs: Dict[str, Message]) -> Union[Message, None]:
        return input_msgs['frame']

    @abstractmethod
    def draw(self, frame_msg: FrameMessage) -> np.ndarray:
        """Draw on the frame image with information from the single frame.

        Args:
            frame_meg (FrameMessage): The frame to get information from and
                draw on.

        Returns:
            array: The output image
        """
