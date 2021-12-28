# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
from queue import Full, Queue
from threading import Thread
from typing import List, Optional, Union

import cv2
import numpy as np
from mmcv import color_val

from mmpose.utils.timer import RunningAverage
from .builder import NODES
from .node import Node

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


@NODES.register_module()
class ModelResultBindingNode(Node):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 result_buffer: str,
                 output_buffer: Union[str, List[str]],
                 synchronous: bool = False):
        super().__init__(name=name, enable=True)
        self.synchronous = synchronous

        # Cache the latest model result
        self.last_result_msg = None

        # Inference speed analysis
        self.result_fps = RunningAverage(window=10)
        self.frame_lag = RunningAverage(window=10)
        self.result_lag = RunningAverage(window=10)

        # Register buffers
        if self.synchronous:
            self.register_input_buffer(result_buffer, 'result', essential=True)
        else:
            self.register_input_buffer(frame_buffer, 'frame', essential=True)
            self.register_input_buffer(result_buffer, 'result')
        self.register_output_buffer(output_buffer)

    def process(self, input_msgs):
        result_msg = input_msgs['result']

        # Update last result
        if result_msg is not None:
            # Update result FPS
            if self.last_result_msg is not None:
                fps = 1.0 / (
                    result_msg.timestamp - self.last_result_msg.timestamp)
                self.result_fps.update(fps)
            # Update inference latency
            self.result_lag.update(time.time() - result_msg.timestamp)
            # Update last inference result
            self.last_result_msg = result_msg

        if not self.synchronous:
            # Asynchronous mode: Bind the latest result with the current frame.
            frame_msg = input_msgs['frame']

            self.frame_lag.update(time.time() - frame_msg.timestamp)

            # Bind result to frame
            if self.last_result_msg is not None:
                frame_msg.set_full_results(
                    self.last_result_msg.get_full_results())
                frame_msg.merge_route_info(
                    self.last_result_msg.get_route_info())

            output_msg = frame_msg

        else:
            # Synchronous mode: Directly output the frame that the model result
            # was obtained from.
            self.frame_lag.update(time.time() - result_msg.timestamp)
            output_msg = result_msg

        return output_msg

    def _get_node_info(self):
        info = super()._get_node_info()
        info['result_fps'] = self.result_fps.average()
        info['result_lag (ms)'] = self.result_lag.average() * 1000
        info['frame_lag (ms)'] = self.frame_lag.average() * 1000
        return info


@NODES.register_module()
class MonitorNode(Node):

    _default_ignore_items = ['timestamp']

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = False,
                 x_offset=20,
                 y_offset=20,
                 y_delta=15,
                 text_color='black',
                 background_color=(255, 183, 0),
                 text_scale=0.4,
                 style='simple',
                 ignore_items: Optional[List[str]] = None):
        super().__init__(name=name, enable_key=enable_key, enable=enable)

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.y_delta = y_delta
        self.text_color = color_val(text_color)
        self.background_color = color_val(background_color)
        self.text_scale = text_scale
        self.style = style
        if ignore_items is None:
            self.ignore_items = self._default_ignore_items
        else:
            self.ignore_items = ignore_items

        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        self.register_output_buffer(output_buffer)

    def process(self, input_msgs):
        frame_msg = input_msgs['frame']

        frame_msg.update_route_info(
            node_name='System Info',
            node_type='dummy',
            info=self._get_system_info())

        img = frame_msg.get_image()
        route_info = frame_msg.get_route_info()
        img = self._show_route_info(img, route_info)

        frame_msg.set_image(img)
        return frame_msg

    def _get_system_info(self):
        sys_info = {}
        if psutil_proc is not None:
            sys_info['CPU(%)'] = psutil_proc.cpu_percent()
            sys_info['Memory(%)'] = psutil_proc.memory_percent()
        return sys_info

    def _show_route_info(self, img, route_info):
        if self.style == 'fancy':
            return self._show_route_info_fancy(img, route_info)
        else:
            return self._show_route_info_simple(img, route_info)

    def _show_route_info_simple(self, img, route_info):

        x = self.x_offset
        y = self.y_offset

        def _put_line(line=''):
            nonlocal y
            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        self.text_scale, self.text_color, 1)
            y += self.y_delta

        for node_info in route_info:
            title = f'{node_info["node"]}({node_info["node_type"]})'
            _put_line(title)
            for k, v in node_info['info'].items():
                if k in self.ignore_items:
                    continue
                if isinstance(v, float):
                    v = f'{v:.1f}'
                _put_line(f'    {k}: {v}')

        return img

    def _show_route_info_fancy(self, img, route_info):
        canvas = np.full(img.shape, self.background_color, dtype=img.dtype)

        x = self.x_offset
        y = self.y_offset

        max_len = 0

        def _put_line(line=''):
            nonlocal y, max_len
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        self.text_scale, self.text_color, 1)
            y += self.y_delta
            max_len = max(max_len, len(line))

        for node_info in route_info:
            title = f'{node_info["node"]}({node_info["node_type"]})'
            _put_line(title)
            for k, v in node_info['info'].items():
                if k in self.ignore_items:
                    continue
                if isinstance(v, float):
                    v = f'{v:.1f}'
                _put_line(f'    {k}: {v}')

        x1 = max(0, self.x_offset)
        x2 = min(img.shape[1], int(x + max_len * self.text_scale * 20))
        y1 = max(0, self.y_offset - self.y_delta)
        y2 = min(img.shape[0], y)

        src1 = canvas[y1:y2, x1:x2]
        src2 = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

        return img

    def bypass(self, input_msgs):
        return input_msgs['frame']


@NODES.register_module()
class RecorderNode(Node):
    """Record the frames into a local file."""

    def __init__(
        self,
        name: str,
        frame_buffer: str,
        output_buffer: Union[str, List[str]],
        out_video_file: str,
        out_video_fps: int = 30,
        out_video_codec: str = 'mp4v',
        buffer_size: int = 30,
    ):
        super().__init__(name=name, enable_key=None, enable=True, daemon=False)

        self.queue = Queue(maxsize=buffer_size)
        self.out_video_file = out_video_file
        self.out_video_fps = out_video_fps
        self.out_video_codec = out_video_codec
        self.vwriter = None

        # Register buffers
        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        self.register_output_buffer(output_buffer)

        # Start a new thread to write frame
        self.t_record = Thread(target=self._record, args=(), daemon=True)
        self.t_record.start()

    def process(self, input_msgs):

        frame_msg = input_msgs['frame']
        img = frame_msg.get_image() if frame_msg is not None else None
        img_queued = False

        while not img_queued:
            try:
                self.queue.put(img, timeout=1)
                img_queued = True
                logging.info(f'{self.name}: recorder received one frame!')
            except Full:
                logging.warn(f'{self.name}: recorder jamed!')

        return frame_msg

    def _record(self):

        while True:

            img = self.queue.get()

            if img is None:
                break

            if self.vwriter is None:
                fourcc = cv2.VideoWriter_fourcc(*self.out_video_codec)
                fps = self.out_video_fps
                frame_size = (img.shape[1], img.shape[0])
                self.vwriter = cv2.VideoWriter(self.out_video_file, fourcc,
                                               fps, frame_size)
                assert self.vwriter.isOpened()

            self.vwriter.write(img)

        logging.warn('Video recorder released!')
        if self.vwriter is not None:
            self.vwriter.release()

    def on_exit(self):
        try:
            # Try putting a None into the output queue so the self.vwriter will
            # be released after all queue frames have been written to file.
            self.queue.put(None, timeout=1)
            self.t_record.join(timeout=1)
        except Full:
            pass

        if self.t_record.is_alive():
            # Force to release self.vwriter
            logging.warn('Video recorder forced release!')
            if self.vwriter is not None:
                self.vwriter.release()
