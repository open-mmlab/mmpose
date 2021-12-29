# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from queue import Empty
from threading import Thread
from typing import Callable, Dict, List, Optional, Tuple, Union

from mmcv.utils.misc import is_method_overridden

from mmpose.utils import StopWatch
from ..utils import Message, VideoEndingMessage, limit_max_fps


@dataclass
class BufferInfo():
    """Dataclass for buffer information."""
    buffer_name: str
    input_name: Optional[str] = None
    essential: bool = False


@dataclass
class EventInfo():
    """Dataclass for event handler information."""
    event_name: str
    is_keyboard: bool = False
    handler_func: Optional[Callable] = None


class Node(Thread, metaclass=ABCMeta):
    """Base interface of functional module.

    Parameters:
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
        enable (bool): Default enable/disable status. Default: True.
        daemon (bool): Whether node is a daemon. Default: True.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 enable_key: Optional[Union[str, int]] = None,
                 max_fps: int = 30,
                 input_check_interval: float = 0.01,
                 enable: bool = True,
                 daemon=False):
        super().__init__(name=name, daemon=daemon)
        self._runner = None
        self._enabled = enable
        self.enable_key = enable_key
        self.max_fps = max_fps
        self.input_check_interval = input_check_interval

        # A partitioned buffer manager the runner's buffer manager that
        # only accesses the buffers related to the node
        self._buffer_manager = None

        # Input/output buffers are a list of registered buffers' information
        self._input_buffers = []
        self._output_buffers = []

        # Event manager is a copy of assigned runner's event manager
        self._event_manager = None

        # A list of registered event information
        # See register_event() for more information
        # Note that we recommend to handle events in nodes by registering
        # handlers, but one can still access the raw event by _event_manager
        self._registered_events = []

        # A list of (listener_threads, event_info)
        # See set_runner() for more information
        self._event_listener_threads = []

        # A timer to calculate node FPS
        self._timer = StopWatch(window=10)

        # Register enable toggle key
        if self.enable_key:
            # If the node allows toggling enable, it should override the
            # `bypass` method to define the node behavior when disabled.
            if not is_method_overridden('bypass', Node, self.__class__):
                raise NotImplementedError(
                    f'The node {self.__class__} does not support toggling'
                    'enable but got argument `enable_key`. To support toggling'
                    'enable, please override the `bypass` method of the node.')

            self.register_event(
                event_name=self.enable_key,
                is_keyboard=True,
                handler_func=self._toggle_enable,
            )

    @property
    def registered_buffers(self):
        return self._input_buffers + self._output_buffers

    @property
    def registered_events(self):
        return self._registered_events.copy()

    def _toggle_enable(self):
        self._enabled = not self._enabled

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
        buffer_info = BufferInfo(buffer_name, input_name, essential)
        self._input_buffers.append(buffer_info)

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
            buffer_info = BufferInfo(name)
            self._output_buffers.append(buffer_info)

    def register_event(self,
                       event_name: str,
                       is_keyboard: bool = False,
                       handler_func: Optional[Callable] = None):
        """Register an event. All events used in the node need to be registered
        in __init__(). If a callable handler is given, a thread will be create
        to listen and handle the event when the node starts.

        Args:
            Args:
            event_name (str|int): The event name. If is_keyboard==True,
                event_name should be a str (as char) or an int (as ascii)
            is_keyboard (bool): Indicate whether it is an keyboard
                event. If True, the argument event_name will be regarded as a
                key indicator.
            handler_func (callable, optional): The event handler function,
                which should be a collable object with no arguments or
                return values. Default: None.
        """
        event_info = EventInfo(event_name, is_keyboard, handler_func)
        self._registered_events.append(event_info)

    def set_runner(self, runner):
        # Get partitioned buffer manager
        buffer_names = [
            buffer.buffer_name
            for buffer in self._input_buffers + self._output_buffers
        ]
        self._buffer_manager = runner.buffer_manager.get_sub_manager(
            buffer_names)

        # Get event manager
        self._event_manager = runner.event_manager

    def _get_input_from_buffer(self) -> Tuple[bool, Optional[Dict]]:
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
        buffer_manager = self._buffer_manager

        if buffer_manager is None:
            raise ValueError(f'{self.name}: Runner not set!')

        # Check that essential buffers are ready
        for buffer_info in self._input_buffers:
            if buffer_info.essential and buffer_manager.is_empty(
                    buffer_info.buffer_name):
                return False, None

        # Default input
        result = {
            buffer_info.input_name: None
            for buffer_info in self._input_buffers
        }

        for buffer_info in self._input_buffers:
            try:
                result[buffer_info.input_name] = buffer_manager.get(
                    buffer_info.buffer_name, block=False)
            except Empty:
                if buffer_info.essential:
                    # Return unsuccessful flag if any
                    # essential input is unready
                    return False, None

        return True, result

    def _send_output_to_buffers(self, output_msg):
        """Send output of the process method to registered output buffers.

        Args:
            output_msg (Message): output message
            force (bool, optional): If True, block until the output message
                has been put into all output buffers. Default: False
        """
        for buffer_info in self._output_buffers:
            buffer_name = buffer_info.buffer_name
            self._buffer_manager.put_force(buffer_name, output_msg)

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

        logging.info(f'Node {self.name} starts')

        # Create event listener threads
        for event_info in self._registered_events:

            if event_info.handler_func is None:
                continue

            def event_listener():
                while True:
                    with self._event_manager.wait_and_handle(
                            event_info.event_name, event_info.is_keyboard):
                        event_info.handler_func()

            t_listener = Thread(target=event_listener, args=(), daemon=True)
            t_listener.start()
            self._event_listener_threads.append(t_listener)

        # Loop
        while True:
            # Exit
            if self._event_manager.is_set('_exit_'):
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
            video_ending = False
            for _, msg in input_msgs.items():
                if isinstance(msg, VideoEndingMessage):
                    self._send_output_to_buffers(msg)
                    video_ending = True
                    break

            if video_ending:
                self.on_exit()
                break

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

        logging.info(f'{self.name}: process ending.')
