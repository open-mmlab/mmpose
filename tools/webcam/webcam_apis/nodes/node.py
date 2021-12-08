# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings
import weakref
from abc import ABCMeta, abstractmethod
from threading import Thread
from typing import Optional, Union

from mmcv.utils.misc import is_method_overridden

from mmpose.utils import StopWatch
from ..utils import Message, limit_max_fps


class Node(Thread, metaclass=ABCMeta):

    def __init__(self,
                 name: Optional[str] = None,
                 enable_key: Optional[Union[str, int]] = None,
                 max_fps: Optional[float] = 30):
        super().__init__(name=name, daemon=True)
        self._runner = None
        self._enabled = True
        self.enable_key = enable_key
        self.input_buffers = []
        self.output_buffers = []
        self.max_fps = max_fps

        # If the node allows toggling enable, it should override the `bypass`
        # method to define the node behavior when disabled.
        if self.enable_key:
            if not is_method_overridden('bypass', Node, self.__class__):
                raise NotImplementedError(
                    f'The node {self.__class__} does not support toggling'
                    'enable but got argument `enable_key`. To support toggling'
                    'enable, please override the `bypass` method of the node.')

        # A timer to calculate node FPS
        self._timer = StopWatch(window=10)

    def register_input_buffer(self, buffer_name, input_name, essential=False):
        buffer_info = {
            'buffer_name': buffer_name,
            'input_name': input_name,
            'essential': essential
        }
        self.input_buffers.append(buffer_info)

    def register_output_buffer(self, buffer_name):
        buffer_info = {'buffer_name': buffer_name}
        self.output_buffers.append(buffer_info)

    @property
    def runner(self):
        if self._runner is None:
            warnings.warn(
                f'Node {self.name} has not been registered to a runner.',
                RuntimeWarning)
            return None
        return self._runner()

    def set_runner(self, runner):
        if self._runner is not None:
            raise RuntimeError(
                f'Node {self.name} has been registered to a runner. '
                'Re-registering is not allowed.')

        self._runner = weakref.ref(runner)

    def check_enabled(self):
        # Always enabled if there is no toggle key
        if self.enable_key is None:
            return True

        if self.runner.event_manager.is_set_keyboard(self.enable_key):
            self._enabled = not self._enabled
            self.runner.event_manager.clear_keyboard(self.enable_key)

        return self._enabled

    def get_input(self) -> tuple[bool, Optional[dict]]:
        """Get and pack input data if it's ready. The function returns a tuple
        of a status flag and a packed data dictionary. If input_buffer is
        ready, the status flag will be True, and the packed data is a dict
        whose items are buffer names and corresponding messages (unready
        additional buffers will give a `None`). Otherwise, the status flag is
        False and the packed data is None.

        Returns:
            bool: status flag
            dict[str, Message]: the packed inputs where the key is the buffer
                name and the value is the Message got from the corresponding
                buffer.
        """
        buffer_manager = self.runner.buffer_manager

        # Check that essential buffers are ready
        for buffer_info in self.input_buffers:
            if buffer_info['essential'] and buffer_manager.is_empty(
                    buffer_info['buffer_name']):
                return False, None

        result = {}
        for buffer_info in self.input_buffers:
            try:
                data = buffer_manager.get(buffer_info['buffer_name'])
            except (IndexError, KeyError):
                if buffer_info['essential']:
                    return False, None
                data = None

            result[buffer_info['input_name']] = data

        return True, result

    @abstractmethod
    def process(self, input_msgs: dict[str, Message]) -> Union[Message, None]:
        ...

    def bypass(self, input_msgs: dict[str, Message]) -> Union[Message, None]:
        raise NotImplementedError

    def get_node_info(self):
        info = {'fps': self._timer.report('_FPS_'), 'timestamp': time.time()}
        return info

    def run(self):

        while True:

            # Check if input is ready
            input_status, input_msgs = self.get_input()
            if not input_status:
                time.sleep(0.001)
                continue

            # Check if enabled
            enabled = self.check_enabled()
            if not enabled:
                # Override bypass method to define node behavior when disabled
                output_msg = self.bypass(input_msgs)
            else:
                with self._timer.timeit():
                    with limit_max_fps(self.max_fps):
                        # Process
                        output_msg = self.process(input_msgs)

                if output_msg:
                    # Update route information
                    node_info = self.get_node_info()
                    output_msg.update_route_info(node=self, info=node_info)

            # Dispatch output message
            if output_msg:
                for buffer_info in self.output_buffers:
                    buffer_name = buffer_info['buffer_name']
                    self.runner.buffer_manager.put(buffer_name, output_msg)
