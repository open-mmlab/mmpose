# Copyright (c) OpenMMLab. All rights reserved.
import time
from threading import Thread
from typing import Optional

import cv2

from .nodes import NODES
from .utils import BufferManager, EventManager, FrameMessage, limit_max_fps


class WebcamRunner():

    def __init__(self,
                 name: str = 'Default Webcam Runner',
                 camera_id: int = 0,
                 camera_fps: int = 30,
                 ms_display_delay: int = 0,
                 user_buffers: Optional[list[tuple[str, int]]] = None,
                 nodes: Optional[list[dict]] = None):

        # Collect arguments
        self.name = name
        self.camera_id = camera_id
        self.camera_fps = camera_fps
        self.ms_display_delay = ms_display_delay

        self.buffer_manager = BufferManager()
        self.event_manager = EventManager()
        self.node_list = []
        self.vcap = None  # Video Capture

        # register default buffers
        self.buffer_manager.add_buffer('_frame_', maxlen=20)
        self.buffer_manager.add_buffer('_input_')
        self.buffer_manager.add_buffer('_display_')

        # register user defined buffers
        if user_buffers:
            for buffer_name, buffer_size in user_buffers:
                self.buffer_manager.add_buffer(buffer_name, buffer_size)

        # register nodes
        if not nodes:
            raise ValueError('No node is registered to the runner.')
        for node_cfg in nodes:
            node = NODES.build(node_cfg)
            node.set_runner(self)
            for buffer_info in node.input_buffers:
                assert self.buffer_manager.has_buffer(
                    buffer_info['buffer_name'])
            for buffer_info in node.output_buffers:
                assert self.buffer_manager.has_buffer(
                    buffer_info['buffer_name'])

            self.node_list.append(node)

    def _read_camera(self):

        print('read_camera thread starts')

        camera_id = self.camera_id
        fps = self.camera_fps

        self.vcap = cv2.VideoCapture(camera_id)

        if not self.vcap.isOpened():
            self.logger.warn(f'Cannot open camera (ID={camera_id})')
            exit()

        while not self.event_manager.is_set('exit'):
            with limit_max_fps(fps):
                # capture a camera frame
                ret_val, frame = self.vcap.read()
                if ret_val:
                    # Put frame message (for display) into buffer
                    frame_msg = FrameMessage(frame)
                    self.buffer_manager.put('_frame_', frame_msg)

                    # Put input message (for model inference or other usage)
                    # into buffer
                    input_msg = FrameMessage(frame)
                    input_msg.update_route_info(
                        node_name='Camera Info',
                        node_type='dummy',
                        info=self.get_camera_info())
                    self.buffer_manager.put('_input_', input_msg)

                else:
                    self.buffer_manager.put('_frame_', None)

        self.vcap.release()

    def _display(self):

        print('display thread starts')

        output_msg = None
        vwriter = None

        while not self.event_manager.is_set('exit'):
            while self.buffer_manager.is_empty('_display_'):
                time.sleep(0.001)

            # acquire output from buffer
            output_msg = self.buffer_manager.get('_display_')

            # None indicates input stream ends
            if output_msg is None:
                self.event_manager.set('exit')
                break

            img = output_msg.get_image()

            # delay control
            if self.ms_display_delay > 0:
                t_sleep = self.ms_display_delay * 0.001 - (
                    time.time() - output_msg.timestamp)
                if t_sleep > 0:
                    time.sleep(t_sleep)

            # show in a window
            cv2.imshow(self.name, img)

            # handle keyboard input
            key = cv2.waitKey(1)
            if key != -1:
                self.on_keyboard_input(key)

        cv2.destroyAllWindows()
        if vwriter is not None:
            vwriter.release()

    def on_keyboard_input(self, key):
        if key in (27, ord('q'), ord('Q')):
            self.event_manager.set('exit')
        else:
            self.event_manager.set_keyboard(key)

    def get_camera_info(self):
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
        print('run')
        try:
            t_read = Thread(target=self._read_camera, args=())
            t_read.start()

            for node in self.node_list:
                node.start()

            # run display in the main thread
            self._display()

            # joint non-daemon threads
            t_read.join()

        except KeyboardInterrupt:
            pass
