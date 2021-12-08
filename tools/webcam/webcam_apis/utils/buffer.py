# Copyright (c) OpenMMLab. All rights reserved.
from collections import deque, namedtuple
from threading import Lock

Buffer = namedtuple('Buffer', ['queue', 'mutex'])


class BufferManager():

    def __init__(self):
        self._buffers = {}

    def add_buffer(self, name, maxlen=1):
        if name in self._buffers:
            raise ValueError('Fail to register new buffer. '
                             f'The name "{name}" has been used.')
        queue = deque(maxlen=maxlen)
        mutex = Lock()
        self._buffers[name] = Buffer(queue, mutex)

    def put(self, name, obj):
        if name not in self._buffers:
            raise ValueError('Fail to put object into buffer. '
                             f'Invalid buffer name "{name}".')

        with self._buffers[name].mutex:
            self._buffers[name].queue.append(obj)

    def get(self, name):
        if name not in self._buffers:
            raise ValueError('Fail to get object from buffer. '
                             f'Invalid queue name "{name}".')

        with self._buffers[name].mutex:
            return self._buffers[name].queue.popleft()

    def delete_buffer(self, name):
        if name in self._buffers:
            _ = self._buffers.pop(name)

    def has_buffer(self, name):
        return name in self._buffers

    def is_empty(self, name):
        return not len(self._buffers[name].queue)
