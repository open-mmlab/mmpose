# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
from queue import Queue
from threading import Thread

import cv2

from .builder import NODES
from .webcam_utils import BufferManager, EventManager, Message


class WebcamRunner():

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.buffer_manager = BufferManager()
        self.event_manager = EventManager()
        self.frame_buffer = Queue(maxsize=cfgs.frame_buffer_size)
        self.node_list = []
        self.logger = logging.getLogger()

        # register default buffers
        self.buffer_manager.add_buffer('_frame', maxlen=cfgs.frame_buffer_size)
        self.buffer_manager.add_buffer('_input')
        self.buffer_manager.add_buffer('_output')

        # register user defined buffers
        for buffer_name, buffer_size in cfgs.buffers:
            self.buffer_manager.add_buffer(buffer_name, buffer_size)

        # register nodes
        for node_cfg in cfgs.nodes:
            node = NODES.build(node_cfg)
            for buffer_info in node.input_buffers:
                assert self.buffer_manager.has_buffer(buffer_info['name'])
            for buffer_info in node.output_buffers:
                assert self.buffer_manager.has_buffer(buffer_info['name'])

            self.node_list.append(node)

    def _read_camera(self):

        self.logger.info('read_camera thread starts')

        cfgs = self.cfgs
        camera_id = cfgs.camera_id

        vcap = cv2.VideoCapture(camera_id)
        if not vcap.isOpened():
            self.logger.warn(f'Cannot open camera (ID={camera_id})')
            exit()

        while not self.event_manager.is_set('exit'):
            # capture a camera frame
            ret_val, frame = vcap.read()
            if ret_val:
                frame_msg = Message(data=frame)
                self.buffer_manager.put('_input', frame_msg)
                self.buffer_manager.put('_frame', frame_msg)
                # self.frame_buffer.put(frame_msg)

            else:
                self.frame_buffer.put(None)

        vcap.release()

    def _display(self):

        self.logger.info('display thread starts')

        output_msg = None
        vwriter = None

        while not self.event_manager.is_set('exit'):
            while self.buffer_manager.is_empty('_output'):
                time.sleep(0.001)

            # acquire output from buffer
            output_msg = self.buffer_manager.get('_output')

            # None indicates input stream ends
            if output_msg is None:
                self.event_manager.set('exit')
                break

            img = output_msg.data

            # delay control
            if self.cfgs.display_delay > 0:
                t_sleep = self.cfgs.display_delay * 0.001 - (
                    time.time() - output_msg.timestamp)
                if t_sleep > 0:
                    time.sleep(t_sleep)

            # show in a window
            cv2.imshow(self.cfgs.name, img)

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

    def run(self):

        try:
            t_read = Thread(target=self._read_camera, args=())
            t_read.start()

            # run display in the main thread
            self._display()

            # joint non-daemon threads
            t_read.join()

        except KeyboardInterrupt:
            pass
