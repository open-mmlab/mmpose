# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
from abc import abstractmethod
from functools import partial
from multiprocessing import Condition, Pool, Value
from multiprocessing.dummy import Condition as ThreadCondition
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.dummy import Value as ThreadValue
from queue import Queue
from threading import Thread
from typing import Callable, Optional, Tuple, Union

from ..utils import Message, VideoEndingMessage, limit_max_fps
from .node import Node

__all__ = ['AsyncNode']


class DummyPool:

    def __init__(self, processes, initializer, initargs):
        self.processes = processes
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def apply_async(self, func, func_args, callback=None, **kwargs):
        res = func(*func_args)
        if callback:
            callback(res)
        return DummyResult(res)


class DummyResult:

    def __init__(self, res):
        self.res = res

    def get(self, *args, **kwargs):
        return self.res


num_available_workers = None


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
        global num_available_workers

        self.num_workers = num_workers if backend != 'dummy' else 1
        self.worker_timeout = worker_timeout
        self.backend = backend

        if self.backend == 'process':
            num_available_workers = Value('i', self.num_workers)
            self.has_available_worker = Condition()
        else:
            num_available_workers = ThreadValue('i', self.num_workers)
            self.has_available_worker = ThreadCondition()

        # Create a queue to store outputs of the worker pool
        self.res_queue = Queue(maxsize=1)

    def run(self):
        global num_available_workers

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
        callback = self.get_callback_func()
        if initializer is not None:
            initargs = self.get_pool_initargs()
        else:
            initargs = ()

        if self.backend == 'dummy':
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

                # Block until a worker is available
                with self.has_available_worker:
                    while num_available_workers.value < 0:
                        self.has_available_worker.wait()

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
                    res = pool.apply_async(
                        self.get_pool_bypass_func(), (input_msgs, ),
                        callback=callback)
                    num_available_workers.value -= 1
                else:
                    with limit_max_fps(self.max_fps):
                        # Process
                        res = pool.apply_async(
                            self.get_pool_process_func(), (input_msgs, ),
                            callback=callback)
                        num_available_workers.value -= 1

                self.res_queue.put(res)

            self.res_queue.put(None)
            t_collect.join()

    def process_pool_output(self, res) -> Optional[Message]:
        """This method is for processing the output of the worker pool."""
        return res.get()

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

    def get_callback_func(self) -> Callable:

        def _callback(res, has_available_workers):
            global num_available_workers
            with has_available_workers:
                num_available_workers.value += 1
                has_available_workers.notify()

        return partial(
            _callback, has_available_workers=self.has_available_worker)

    # Implement a dummy process() to avoid TypeError caused by abstractmethod
    def process(self, input_msgs):
        raise NotImplementedError()
