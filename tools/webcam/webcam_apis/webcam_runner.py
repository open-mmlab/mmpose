# Copyright (c) OpenMMLab. All rights reserved.
import logging
import sys
import time
import warnings
from contextlib import nullcontext
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import cv2

from .nodes import NODES
from .utils import (BufferManager, EventManager, FrameMessage, ImageCapture,
                    VideoEndingMessage, is_image_file, limit_max_fps)

DEFAULT_FRAME_BUFFER_SIZE = 1
DEFAULT_INPUT_BUFFER_SIZE = 1
DEFAULT_DISPLAY_BUFFER_SIZE = 0
DEFAULT_USER_BUFFER_SIZE = 1


class WebcamRunner():
    """An interface for building webcam application from config.

    Parameters:
        name (str): Runner name.
        camera_id (int | str): The camera ID (usually the ID of the default
            camera is 0). Alternatively a file path or a URL can be given
            to load from a video or image file.
        camera_frame_shape (tuple, optional): Set the frame shape of the
            camera in (width, height). If not given, the default frame shape
            will be used. This argument is only valid when using a camera
            as the input source. Default: None
        camera_fps (int): Video reading maximum FPS. Default: 30
        buffer_sizes (dict, optional): A dict to specify buffer sizes. The
            key is the buffer name and the value is the buffer size.
            Default: None
        nodes (list): Node configs.
    """

    def __init__(self,
                 name: str = 'Default Webcam Runner',
                 camera_id: Union[int, str] = 0,
                 camera_fps: int = 30,
                 camera_frame_shape: Optional[Tuple[int, int]] = None,
                 synchronous: bool = False,
                 buffer_sizes: Optional[Dict[str, int]] = None,
                 nodes: Optional[List[Dict]] = None):

        # Basic parameters
        self.name = name
        self.camera_id = camera_id
        self.camera_fps = camera_fps
        self.camera_frame_shape = camera_frame_shape
        self.synchronous = synchronous

        # self.buffer_manager manages data flow between runner and nodes
        self.buffer_manager = BufferManager()
        # self.event_manager manages event-based asynchronous communication
        self.event_manager = EventManager()
        # self.node_list holds all node instance
        self.node_list = []
        # self.vcap is used to read camera frames. It will be built when the
        # runner starts running
        self.vcap = None

        # Register runner events
        self.event_manager.register_event('_exit_', is_keyboard=False)
        if self.synchronous:
            self.event_manager.register_event('_idle_', is_keyboard=False)

        # Register nodes
        if not nodes:
            raise ValueError('No node is registered to the runner.')

        # Register default buffers
        if buffer_sizes is None:
            buffer_sizes = {}
        # _frame_ buffer
        frame_buffer_size = buffer_sizes.get('_frame_',
                                             DEFAULT_FRAME_BUFFER_SIZE)
        self.buffer_manager.register_buffer('_frame_', frame_buffer_size)
        # _input_ buffer
        input_buffer_size = buffer_sizes.get('_input_',
                                             DEFAULT_INPUT_BUFFER_SIZE)
        self.buffer_manager.register_buffer('_input_', input_buffer_size)
        # _display_ buffer
        display_buffer_size = buffer_sizes.get('_display_',
                                               DEFAULT_DISPLAY_BUFFER_SIZE)
        self.buffer_manager.register_buffer('_display_', display_buffer_size)

        # Build all nodes:
        for node_cfg in nodes:
            logging.info(f'Create node: {node_cfg.name}({node_cfg.type})')
            node = NODES.build(node_cfg)

            # Register node
            self.node_list.append(node)

            # Register buffers
            for buffer_info in node.registered_buffers:
                buffer_name = buffer_info.buffer_name
                if buffer_name in self.buffer_manager:
                    continue
                buffer_size = buffer_sizes.get(buffer_name,
                                               DEFAULT_USER_BUFFER_SIZE)
                self.buffer_manager.register_buffer(buffer_name, buffer_size)
                logging.info(
                    f'Register user buffer: {buffer_name}({buffer_size})')

            # Register events
            for event_info in node.registered_events:
                self.event_manager.register_event(
                    event_name=event_info.event_name,
                    is_keyboard=event_info.is_keyboard)
                logging.info(f'Register event: {event_info.event_name}')

        # Set runner for nodes
        # This step is performed after node building when the runner has
        # create full buffer/event managers and can
        for node in self.node_list:
            logging.info(f'Set runner for node: {node.name})')
            node.set_runner(self)

    def _read_camera(self):
        """Continually read video frames and put them into buffers."""

        camera_id = self.camera_id
        fps = self.camera_fps

        # Build video capture
        if is_image_file(camera_id):
            self.vcap = ImageCapture(camera_id)
        else:
            self.vcap = cv2.VideoCapture(camera_id)
            if self.camera_frame_shape is not None:
                width, height = self.camera_frame_shape
                self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.vcap.isOpened():
            warnings.warn(f'Cannot open camera (ID={camera_id})')
            sys.exit()

        # Read video frames in a loop
        first_frame = True
        while not self.event_manager.is_set('_exit_'):
            if self.synchronous:
                if first_frame:
                    cm = nullcontext()
                else:
                    # Read a new frame until the last frame has been processed
                    cm = self.event_manager.wait_and_handle('_idle_')
            else:
                # Read frames with a maximum FPS
                cm = limit_max_fps(fps)

            first_frame = False

            with cm:
                # Read a frame
                ret_val, frame = self.vcap.read()
                if ret_val:
                    # Put frame message (for display) into buffer `_frame_`
                    frame_msg = FrameMessage(frame)
                    self.buffer_manager.put('_frame_', frame_msg)

                    # Put input message (for model inference or other use)
                    # into buffer `_input_`
                    input_msg = FrameMessage(frame.copy())
                    input_msg.update_route_info(
                        node_name='Camera Info',
                        node_type='dummy',
                        info=self._get_camera_info())
                    self.buffer_manager.put_force('_input_', input_msg)

                else:
                    # Put a video ending signal
                    self.buffer_manager.put('_frame_', VideoEndingMessage())

        self.vcap.release()

    def _display(self):
        """Continually obtain and display output frames."""

        output_msg = None

        while not self.event_manager.is_set('_exit_'):
            while self.buffer_manager.is_empty('_display_'):
                time.sleep(0.001)

            # Set _idle_ to allow reading next frame
            if self.synchronous:
                self.event_manager.set('_idle_')

            # acquire output from buffer
            output_msg = self.buffer_manager.get('_display_')

            # None indicates input stream ends
            if isinstance(output_msg, VideoEndingMessage):
                self.event_manager.set('_exit_')
                break

            img = output_msg.get_image()

            # show in a window
            cv2.imshow(self.name, img)

            # handle keyboard input
            key = cv2.waitKey(1)
            if key != -1:
                self._on_keyboard_input(key)

        cv2.destroyAllWindows()

    def _on_keyboard_input(self, key):
        """Handle the keyboard input."""

        if key in (27, ord('q'), ord('Q')):
            logging.info(f'Exit event captured: {key}')
            self.event_manager.set('_exit_')
        else:
            logging.info(f'Keyboard event captured: {key}')
            self.event_manager.set(key, is_keyboard=True)

    def _get_camera_info(self):
        """Return the camera information in a dict."""

        frame_width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_rate = self.vcap.get(cv2.CAP_PROP_FPS)

        cam_info = {
            'Camera ID': self.camera_id,
            'Source resolution': f'{frame_width}x{frame_height}',
            'Source FPS': frame_rate,
        }

        return cam_info

    def run(self):
        """Program entry.

        This method starts all nodes as well as video I/O in separate threads.
        """

        try:
            # Start node threads
            non_daemon_nodes = []
            for node in self.node_list:
                node.start()
                if not node.daemon:
                    non_daemon_nodes.append(node)

            # Create a thread to read video frames
            t_read = Thread(target=self._read_camera, args=())
            t_read.start()

            # Run display in the main thread
            self._display()
            logging.info('Display shut down')

            # joint non-daemon nodes and runner threads
            logging.info('Camera reading about to join')
            t_read.join()

            for node in non_daemon_nodes:
                logging.info(f'Node {node.name} about to join')
                node.join()

        except KeyboardInterrupt:
            pass
