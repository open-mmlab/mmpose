# Copyright (c) OpenMMLab. All rights reserved.
import logging
import sys
import time
import warnings
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import cv2

from .nodes import NODES
from .utils import (BufferManager, EventManager, FrameMessage,
                    VideoEndingMessage, limit_max_fps)

FRAME_BUFFER_SIZE = 30
INPUT_BUFFER_SIZE = 30
DISPLAY_BUFFER_SIZE = 0
DEFAULT_USER_BUFFER_SIZE = 0


class WebcamRunner():
    """An interface for building webcam application from config.

    Args:
        name (str): Runner name.
        camera_id (int|str): Webcam ID. Optionally a path can be given
            as the camere_id to load videos from a local file. Default: 0
        camera_fps (int): Video reading maximum FPS. Default: 30
        user_buffers (list, optional): Define a list of user buffers.
            Each buffer should be given as a tuple (buffer_name, buffer_size):
                - buffer_name (str): A unique name for a buffer
                - buffer_size (int): Max length of the buffer
            Default: None (no user buffer).
        nodes (list): Node configs.
    """

    def __init__(self,
                 name: str = 'Default Webcam Runner',
                 camera_id: Union[int, str] = 0,
                 camera_fps: int = 30,
                 user_buffers: Optional[List[Tuple[str, int]]] = None,
                 nodes: Optional[List[Dict]] = None):

        # Basic parameters
        self.name = name
        self.camera_id = camera_id
        self.camera_fps = camera_fps

        # self.buffer_manager manages data flow between runner and nodes
        self.buffer_manager = BufferManager()
        # self.event_manager manages event-based asynchronous communication
        self.event_manager = EventManager()
        # self.node_list holds all node instance
        self.node_list = []
        # self.vcap is used to read camera frames. It will be built when the
        # runner starts running
        self.vcap = None
        # self.logger is used for logging information
        self.logger = logging.getLogger(self.name)

        # Register default buffers
        self.buffer_manager.register_buffer('_frame_', FRAME_BUFFER_SIZE)
        self.buffer_manager.register_buffer('_input_', INPUT_BUFFER_SIZE)
        self.buffer_manager.register_buffer('_display_', DISPLAY_BUFFER_SIZE)

        # Register user-defined buffers
        if user_buffers:
            for buffer in user_buffers:
                if isinstance(buffer, tuple):
                    buffer_name, buffer_size = buffer
                else:
                    buffer_name, buffer_size = buffer, DEFAULT_USER_BUFFER_SIZE
                logging.info(
                    f'Create user buffer: {buffer_name}({buffer_size})')
                self.buffer_manager.register_buffer(buffer_name, buffer_size)
        
        # Register runner events
        self.event_manager.register_event('_exit_', is_keyboard=False)

        # Register nodes
        if not nodes:
            raise ValueError('No node is registered to the runner.')

        # Build all nodes:
        for node_cfg in nodes:
            logging.info(f'Create node: {node_cfg.name}({node_cfg.type})')
            node = NODES.build(node_cfg)
            self.node_list.append(node)

            for buffer_info in node.input_buffers + node.output_buffers:
                assert buffer_info.buffer_name in self.buffer_manager
            for handler_info in node.registered_event_handlers:
                self.event_manager.register_event(event_name=handler_info.event_name, is_keyboard=handler_info.is_keyboard)

        # Register runner for all nodes
        for node in self.node_list:
            logging.info(
                f'Register runner for node: {node_cfg.name}({node_cfg.type})')
            node.set_runner(self)

    def _read_camera(self):
        """Continually read video frames and put them into buffers."""

        camera_id = self.camera_id
        fps = self.camera_fps

        # Build video capture
        self.vcap = cv2.VideoCapture(camera_id)
        if not self.vcap.isOpened():
            warnings.warn(f'Cannot open camera (ID={camera_id})')
            sys.exit()

        # Read video frames in a loop
        while not self.event_manager.is_set('_exit_'):
            with limit_max_fps(fps):
                # Read a frame
                ret_val, frame = self.vcap.read()
                if ret_val:
                    # Put frame message (for display) into buffer `_frame_`
                    frame_msg = FrameMessage(frame)
                    self.buffer_manager.put('_frame_', frame_msg)

                    # Put input message (for model inference or other use)
                    # into buffer `_input_`
                    input_msg = FrameMessage(frame)
                    input_msg.update_route_info(
                        node_name='Camera Info',
                        node_type='dummy',
                        info=self._get_camera_info())
                    self.buffer_manager.try_put('_input_', input_msg)

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

        logging.info(f'Keyboard event captured: {key}')
        if key in (27, ord('q'), ord('Q')):
            self.event_manager.set('_exit_')
        else:
            logging.warn(f'Set keyboard {key}')
            self.event_manager.set(key, is_keyboard=True)

    def _get_camera_info(self):
        """Return the camera information in a dict."""

        frame_width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_rate = self.vcap.get(cv2.CAP_PROP_FPS)

        cam_info = {
            'Camera ID': self.camera_id,
            'Frame size': f'{frame_width}x{frame_height}',
            'Frame rate': frame_rate,
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

            # joint non-daemon nodes and runner threads
            t_read.join()
            for node in non_daemon_nodes:
                node.join()

        except KeyboardInterrupt:
            pass
