# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings
import weakref
from abc import ABCMeta, abstractmethod
from threading import Thread
from typing import Callable, Dict, List, Optional, Tuple, Union

from mmcv.utils.misc import is_method_overridden
from tools.webcam.webcam_apis.utils.message import VideoEndingMessage

from mmpose.utils import StopWatch
from ..utils import Message, limit_max_fps


class Node(Thread, metaclass=ABCMeta):
    """Base interface of functional module.

    Args:
        name (str, optional): The node name (also thread name).
        enable_key (str|int, optional): Set a hot-key to toggle enable/disable
            of the node. If an int value is given, it will be treated as an
            ascii code of a key. Please note:
                1. If enable_key is set, the bypass method need to be
                    overridden to define the node behavior when disabled
                2. Some hot-key has been use for particular use. For example:
                    'q', 'Q' and 27 are used for quit
            Default: None
        max_fps (int): Maximum FPS of the node. This is to avoid the node
            running unrestrictedly and causing large resource consuming.
            Default: 30
        input_check_interval (float): Minimum interval (in millisecond) between
            checking if input is ready. Default: 0.001
    """

    def __init__(self,
                 name: Optional[str] = None,
                 enable_key: Optional[Union[str, int]] = None,
                 max_fps: int = 30,
                 input_check_interval: float = 0.001,
                 daemon=True):
        super().__init__(name=name, daemon=daemon)
        self._runner = None
        self._enabled = True
        self.enable_key = enable_key
        self.input_buffers = []
        self.output_buffers = []
        self.max_fps = max_fps
        self.input_check_interval = input_check_interval

        # A timer to calculate node FPS
        self._timer = StopWatch(window=10)

        # Event listener
        # The key is the event name, and the value contains handler function
        # and corresponding thread
        self._registered_event_handler = {}

        # If the node allows toggling enable, it should override the `bypass`
        # method to define the node behavior when disabled.
        if self.enable_key:
            if not is_method_overridden('bypass', Node, self.__class__):
                raise NotImplementedError(
                    f'The node {self.__class__} does not support toggling'
                    'enable but got argument `enable_key`. To support toggling'
                    'enable, please override the `bypass` method of the node.')

    def register_input_buffer(self,
                              buffer_name: str,
                              input_name: str,
                              essential: bool = False):
        """Register an input buffer, so that Node can automatically check if
        data is ready, fetch data from the buffers and format the inputs to
        feed into `process` method.

        This method can be invoked multiple times to register multiple input
        buffers.

        The subclass of Node should invoke `register_input_buffer` in its
        `__init__` method.

        Args:
            buffer_name (str): The name of the buffer
            input_name (str): The name of the fetched message from the
                corresponding buffer
            essential (bool): An essential input means the node will wait
                until the input is ready before processing. Otherwise, an
                inessential input will not block the processing, instead
                a None will be fetched if the buffer is not ready.
        """
        buffer_info = {
            'buffer_name': buffer_name,
            'input_name': input_name,
            'essential': essential
        }
        self.input_buffers.append(buffer_info)

    def register_output_buffer(self, buffer_name: Union[str, List[str]]):
        """Register one or multiple output buffers, so that the Node can
        automatically send the output of the `process` method to these buffers.

        The subclass of Node should invoke `register_output_buffer` in its
        `__init__` method.

        Args:
            buffer_name (str|list): The name(s) of the output buffer(s).
        """

        if not isinstance(buffer_name, list):
            buffer_name = [buffer_name]

        for name in buffer_name:
            buffer_info = {'buffer_name': name}
            self.output_buffers.append(buffer_info)

    def register_event_handler(self, event_name: str, handler_func: Callable):
        """Register an event handler. A thread will be create to listen and
        handle the event after the runner is set.

        Args:
            event_name(str): The event name.
            handler_func(callable): The event handler function, which should be
                a collable object with no arguments or return values.
        """

        self._registered_event_handler[event_name] = {
            'handler_func': handler_func,
            'keyboard': False,
            'thread': None,
        }

    def register_keyboard_handler(self, key: Optional[Union[str, int]],
                                  handler_func: Callable):
        """Register an keyboard event handler. A thread will be create to
        listen and handle the event after the runner is set.

        Args:
            key(str|int): .
            handler_func(callable): The event handler function, which should be
                a collable object with no arguments or return values.
        """

        self._registered_event_handler[key] = {
            'handler_func': handler_func,
            'keyboard': True,
            'thread': None,
        }

    @property
    def runner(self):
        if self._runner is None:
            warnings.warn(
                f'Node {self.name} has not been registered to a runner, '
                'or the runner has been destroyed.', RuntimeWarning)
            return None
        return self._runner()

    def set_runner(self, runner):
        if self._runner is not None:
            raise RuntimeError(
                f'Node {self.name} has been registered to a runner. '
                'Re-registering is not allowed.')

        self._runner = weakref.ref(runner)

        # Register enable_key
        if self.enable_key:

            def _toggle_enable():
                self._enabled = not self._enabled

            self.register_keyboard_handler(
                key=self.enable_key, handler_func=_toggle_enable)

        # Register all event handler
        for event, info in self._registered_event_handler.items():
            is_keyboard = info['keyboard']
            handler_func = info['handler_func']
            event_manager = self.runner.event_manager

            def event_listener():
                while True:
                    if is_keyboard:
                        event_manager.wait_keyboard(event)
                        handler_func()
                        event_manager.clear_keyboard(event)
                    else:
                        event_manager.wait_keyboard(event)
                        handler_func()
                        event_manager.clear_keyboard(event)

            t_event_listener = Thread(
                target=event_listener, args=(), daemon=True)
            t_event_listener.start()
            info['thread'] = t_event_listener

    def _get_input_from_buffer(self) -> Tuple[bool, Optional[dict]]:
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

        if self.runner is None:
            raise ValueError(f'{self.name}: Runner does not exist!')

        buffer_manager = self.runner.buffer_manager

        # Check that essential buffers are ready
        for buffer_info in self.input_buffers:
            if buffer_info['essential'] and buffer_manager.is_empty(
                    buffer_info['buffer_name']):
                return False, None

        # Default input
        result = {
            buffer_info['input_name']: None
            for buffer_info in self.input_buffers
        }

        for buffer_info in self.input_buffers:
            try:
                data = buffer_manager.get(buffer_info['buffer_name'])
            except (IndexError, KeyError):
                if buffer_info['essential']:
                    return False, None
                data = None

            result[buffer_info['input_name']] = data

        return True, result

    def _send_output_to_buffers(self, output_msg):
        """Send output of the process method to registered output buffers."""
        for buffer_info in self.output_buffers:
            buffer_name = buffer_info['buffer_name']
            self.runner.buffer_manager.put(buffer_name, output_msg)

    @abstractmethod
    def process(self, input_msgs: Dict[str, Message]) -> Union[Message, None]:
        """The core method that implement the function of the node. This method
        will be invoked when the node is enabled and the input data is ready.

        All subclasses of Node should override this method.

        Args:
            input_msgs (dict): The input data collected from the buffers. For
                each item, the key is the `input_name` of the registered input
                buffer, while the value is a Message instance fetched from the
                buffer (or None if the buffer is unessential and not ready).

        Returns:
            Message: The output message of the node. It will be send to all
                registered output buffers.
        """

    def bypass(self, input_msgs: Dict[str, Message]) -> Union[Message, None]:
        """The method that defines the node behavior when disabled. Note that
        if the node has an `enable_key`, this method should be override.

        The method input/output is same as it of `process` method.

        Args:
            input_msgs (dict): The input data collected from the buffers. For
                each item, the key is the `input_name` of the registered input
                buffer, while the value is a Message instance fetched from the
                buffer (or None if the buffer is unessential and not ready).

        Returns:
            Message: The output message of the node. It will be send to all
                registered output buffers.
        """
        raise NotImplementedError

    def _get_node_info(self):
        """Get route information of the node."""
        info = {'fps': self._timer.report('_FPS_'), 'timestamp': time.time()}
        return info

    def on_exit(self):
        """This method will be invoked on event `_exit_`.

        Subclasses should override this method to specifying the exiting
        behavior.
        """

    def run(self):
        """Method representing the Node's activity.

        This method override the standard run() method of Thread. Users should
        not override this method in subclasses.
        """

        while True:
            # Exit
            if self.runner.event_manager.is_set('_exit_'):
                self.on_exit()
                break

            # Check if input is ready
            input_status, input_msgs = self._get_input_from_buffer()

            # Input is not ready
            if not input_status:
                time.sleep(self.input_check_interval)
                continue

            # If a VideoEndingMessage is received, broadcast the signal
            # without invoking process() or bypass()
            for _, msg in input_msgs.items():
                if isinstance(msg, VideoEndingMessage):
                    self._send_output_to_buffers(msg)
                    continue

            # Check if enabled
            if not self._enabled:
                # Override bypass method to define node behavior when disabled
                output_msg = self.bypass(input_msgs)
            else:
                with self._timer.timeit():
                    with limit_max_fps(self.max_fps):
                        # Process
                        output_msg = self.process(input_msgs)

                if output_msg:
                    # Update route information
                    node_info = self._get_node_info()
                    output_msg.update_route_info(node=self, info=node_info)

            # Send output message
            if output_msg is not None:
                self._send_output_to_buffers(output_msg)
