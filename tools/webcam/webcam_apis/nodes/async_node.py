# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
from abc import abstractmethod
from multiprocessing import Pool, TimeoutError
from multiprocessing.pool import AsyncResult, ThreadPool
from queue import Queue
from threading import Thread
from typing import Callable, Optional, Tuple, Union

from tools.webcam.webcam_apis.utils.message import Message

from ..utils import VideoEndingMessage, limit_max_fps
from .node import Node

__all__ = ['AsyncNode']


class DummyPool:

    def __init__(self, num_worker, initializer, initargs):
        assert num_worker == 0
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def apply_async(self, func, func_args, *args, **kwargs):
        res = func(*func_args)
        return DummyResult(res)


class DummyResult:

    def __init__(self, res):
        self.res = res

    def get(self, *args, **kwargs):
        return self.res


class AsyncNode(Node):

    def __init__(self,
                 name: Optional[str] = None,
                 enable_key: Optional[Union[str, int]] = None,
                 max_fps: int = 30,
                 input_check_interval: float = 0.01,
                 enable: bool = True,
                 daemon: bool = False,
                 num_workers: Optional[int] = None,
                 worker_timeout: Optional[float] = None,
                 backend: str = 'thread'):
        super().__init__(
            name=name,
            enable_key=enable_key,
            max_fps=max_fps,
            input_check_interval=input_check_interval,
            enable=enable,
            daemon=daemon)

        self.num_workers = num_workers
        self.worker_timeout = worker_timeout
        self.backend = backend

        # Create a queue to store outputs of the worker pool
        self.res_queue = Queue(maxsize=1)

    def run(self):
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

            t_listener = Thread(target=event_listener, daemon=True)
            t_listener.start()
            self._event_listener_threads.append(t_listener)

        # Create collection thread
        t_collect = Thread(target=self.collect, daemon=False)
        t_collect.start()

        # Build Pool
        initializer = self.get_pool_initializer()
        if initializer is not None:
            initargs = self.get_pool_initargs()
        else:
            initargs = ()

        if self.num_workers == 0:
            _Pool = DummyPool
        elif self.backend == 'thread':
            _Pool = ThreadPool
        elif self.backend == 'process':
            _Pool = Pool
        else:
            raise ValueError(f'{self.name}: Invalid backend {self.backend}')

        with _Pool(self.num_workers, initializer, initargs) as pool:

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
                    # Override bypass method to define node behavior when
                    # disabled
                    pool.apply_async(
                        self.get_pool_bypass_func(), (input_msgs, ),
                        callback=self.res_queue.put)
                else:
                    with limit_max_fps(self.max_fps):
                        # Process
                        pool.apply_async(
                            self.get_pool_process_func(), (input_msgs, ),
                            callback=self.res_queue.put)

                # self.res_queue.put(res)

            self.res_queue.put(None)
            t_collect.join()

    def process_pool_output(self,
                            res: Union[AsyncResult,
                                       DummyResult]) -> Optional[Message]:
        """This method is for processing the output of the worker pool.

        The output of the pool is an AsyncResult instance. The default behavior
        is to get the actual result by AsyncResult.get() with a timeout limit.
        """
        try:
            # return res.get(self.worker_timeout)
            return res
        except TimeoutError:
            logging.warn(f'{self.name}: timeout when getting result. '
                         'Please consider setting a larger'
                         ' `worker_timeout`.')
        return None

    def collect(self):

        while True:
            with self._timer.timeit():
                res = self.res_queue.get()

                if res is None:
                    break

                output_msg = self.process_pool_output(res)

                if output_msg:
                    # Update route information
                    node_info = self._get_node_info()
                    output_msg.update_route_info(node=self, info=node_info)

                    self._send_output_to_buffers(output_msg)

    @abstractmethod
    def get_pool_process_func(self) -> Callable:
        """Get the pool worker function for processing input messages."""

    def get_pool_bypass_func(self) -> Callable:
        """Get the pool workfer function for bypass."""
        raise NotImplementedError

    def get_pool_initializer(self) -> Callable:
        """Get the pool initializer function."""
        return None

    def get_pool_initargs(self) -> Tuple:
        """Get the initializer arguments."""
        return ()

    # Implement a dummy process() to avoid TypeError caused by abstractmethod
    def process(self, input_msgs):
        raise NotImplementedError()
