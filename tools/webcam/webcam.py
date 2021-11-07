# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
import uuid
from queue import Queue
from threading import Thread
from typing import Any, Optional

import cv2
from webcam_utils import BufferManager, EventManager


class Message():

    def __init__(self,
                 msg: str = '',
                 data: Any = None,
                 meta: Any = None,
                 timestamp: Optional[float] = None):
        self.msg = msg
        self.data = data
        self.meta = meta
        self.timestamp = timestamp if timestamp else time.time()
        self.id = uuid.uuid4()


class WebcamRunner():

    def __init__(self, args):
        self.args = args
        self.buffer_manager = BufferManager()
        self.event_manager = EventManager()
        self.frame_buffer = Queue(maxsize=args.frame_buffer_size)
        self.node_list = []
        self.logger = logging.getLogger()

        # register default queues
        self.buffer_manager.add_buffer('input')
        self.buffer_manager.add_buffer('output')

        # register nodes

    def _read_camera(self):

        self.logger.info('read_camera thread starts')

        args = self.args
        camera_id = args.camera_id

        vcap = cv2.VideoCapture(camera_id)
        if not vcap.isOpened():
            self.logger.warn(f'Cannot open camera (ID={camera_id})')
            exit()

        while not self.event_manager.is_set('exit'):
            # capture a camera frame
            ret_val, frame = vcap.read()
            if ret_val:
                frame_msg = Message(data=frame)
                self.buffer_manager.put(frame_msg, 'input')

                if self.args.synchronous_mode:
                    # synchronize input with specific operation (e.g.
                    # model inference accomplished)
                    self.event_manager.wait('synchronize')

                self.frame_buffer.put(frame_msg)

            else:
                self.frame_buffer.put(None)

        vcap.release()

    def _display(self):

        self.logger.info('display thread starts')

        output_msg = None
        vwriter = None

        while not self.event_manager.is_set('exit'):
            # acquire a frame from buffer
            frame_msg = self.frame_buffer.get()
            # None indicates input stream ends
            if frame_msg is None:
                self.event_manager.set('exit')
                break

            img = frame_msg.data

            try:
                output_msg = self.buffer_manager.pop_from('output')
            except IndexError:
                # do not update output_msg
                pass

            # visualize the node outputs
            if output_msg:
                pass

            # delay control
            if self.args.display_delay > 0:
                t_sleep = self.args.display_delay * 0.001 - (
                    time.time() - frame_msg.timestamp)
                if t_sleep > 0:
                    time.sleep(t_sleep)

            # show in a window
            cv2.imshow(self.args.name, img)

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
