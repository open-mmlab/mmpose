# Copyright (c) OpenMMLab. All rights reserved.
import time
import uuid
from collections import defaultdict, deque, namedtuple
from threading import Event, Lock
from typing import Optional

Buffer = namedtuple('Buffer', ['queue', 'mutex'])


class Message():

    def __init__(self,
                 msg: str = '',
                 data: Optional[dict] = None,
                 timestamp: Optional[float] = None):
        self.msg = msg
        self.data = data if data else {}
        self.route_info = []
        self.timestamp = timestamp if timestamp else time.time()
        self.id = uuid.uuid4()

    def update_route_info(self, node, meta=None):
        info = {
            'node': node.name,
            'node_type': node.__class__.__name__,
            'meta': meta
        }
        self.route_info.append(info)


class FrameMessage(Message):

    def __init__(self, img):
        super().__init__(data=dict(image=img))

    def get_image(self):
        return self.data.get('image', None)

    def add_detection_result(self, result, tag=None):
        if 'detection_results' not in self.data:
            self.data['detection_results'] = []
        self.data['detection_results'].append((tag, result))

    def get_detection_result(self, tag=None):
        if 'detection_results' not in self.data:
            return None
        if tag is None:
            results = [res for _, res in self.data['detection_results']]
        else:
            results = [
                res for _tag, res in self.data['detection_results']
                if _tag == tag
            ]
        return results

    def add_pose_result(self, result, tag=None):
        if 'pose_results' not in self.data:
            self.data['pose_results'] = []
        self.data['pose_results'].append((tag, result))

    def get_pose_result(self, tag=None):
        if 'pose_results' not in self.data:
            return None
        if tag is None:
            results = [res for _, res in self.data['pose_results']]
        else:
            results = [
                res for _tag, res in self.data['pose_results'] if _tag == tag
            ]
        return results


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


class EventManager():

    def __init__(self):
        self._events = defaultdict(Event)

    def set(self, name):
        return self._events[name].set()

    def wait(self, name):
        return self._events[name].wait()

    def is_set(self, name):
        return self._events[name].is_set()

    def clear(self, name):
        return self._events[name].clear()

    @staticmethod
    def _get_keyboard_event(key):
        key = ord(key) if isinstance(key, str) else key
        event = f'keyboard_{key}'
        return event

    def set_keyboard(self, key):
        return self.set(self._get_keyboard_event(key))

    def wait_keyboard(self, key):
        return self.wait(self._get_keyboard_event(key))

    def is_set_keyboard(self, key):
        return self.is_set(self._get_keyboard_event(key))

    def clear_keyboard(self, key):
        return self.clear(self._get_keyboard_event(key))
