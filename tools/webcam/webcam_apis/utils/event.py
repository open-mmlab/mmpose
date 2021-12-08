# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from threading import Event


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
