# Copyright (c) OpenMMLab. All rights reserved.
from functools import wraps
from queue import Queue
from typing import Dict, List, Optional

from mmcv import is_seq_of

__all__ = ['BufferManager']


def check_buffer_registered(exist=True):

    def wrapper(func):

        @wraps(func)
        def wrapped(manager, name, *args, **kwargs):
            if exist:
                # Assert buffer exist
                if name not in manager:
                    raise ValueError(f'Fail to call {func.__name__}: '
                                     f'buffer "{name}" is not registered.')
            else:
                # Assert buffer not exist
                if name in manager:
                    raise ValueError(f'Fail to call {func.__name__}: '
                                     f'buffer "{name}" is already registered.')
            return func(manager, name, *args, **kwargs)

        return wrapped

    return wrapper


class Buffer(Queue):

    def put_force(self, item):
        """Force to put an item into the buffer.

        If the buffer is already full, the earliest item in the buffer will be
        remove to make room for the incoming item.
        """
        with self.mutex:
            if self.maxsize > 0:
                while self._qsize() >= self.maxsize:
                    _ = self._get()
                    self.unfinished_tasks -= 1

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class BufferManager():

    def __init__(self,
                 buffer_type: type = Buffer,
                 buffers: Optional[Dict] = None):
        self.buffer_type = buffer_type
        if buffers is None:
            self._buffers = {}
        else:
            if is_seq_of(list(buffers.values()), buffer_type):
                self._buffers = buffers.copy()
            else:
                raise ValueError('The values of buffers should be instance '
                                 f'of {buffer_type}')

    def __contains__(self, name):
        return name in self._buffers

    @check_buffer_registered(False)
    def register_buffer(self, name, maxsize=0):
        self._buffers[name] = self.buffer_type(maxsize)

    @check_buffer_registered()
    def put(self, name, item, block=True, timeout=None):
        self._buffers[name].put(item, block, timeout)

    @check_buffer_registered()
    def put_force(self, name, item):
        self._buffers[name].put_force(item)

    @check_buffer_registered()
    def get(self, name, block=True, timeout=None):
        return self._buffers[name].get(block, timeout)

    @check_buffer_registered()
    def is_empty(self, name):
        return self._buffers[name].empty()

    @check_buffer_registered()
    def is_full(self, name):
        return self._buffers[name].full()

    def get_sub_manager(self, buffer_names: List[str]):
        buffers = {name: self._buffers[name] for name in buffer_names}
        return BufferManager(self.buffer_type, buffers)

    def get_info(self):
        buffer_info = {}
        for name, buffer in self._buffers.items():
            buffer_info[name] = {
                'size': buffer.size,
                'maxsize': buffer.maxsize
            }
        return buffer_info
