# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from contextlib import contextmanager
from threading import Event
from typing import Optional


class EventManager():

    def __init__(self):
        self._events = defaultdict(Event)

    def register_event(self,
                       event_name: str = None,
                       is_keyboard: bool = False):
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        self._events[event_name] = Event()

    def set(self, event_name: str = None, is_keyboard: bool = False):
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        return self._events[event_name].set()

    def wait(self,
             event_name: str = None,
             is_keyboard: Optional[bool] = False,
             timeout: Optional[float] = None):
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        return self._events[event_name].wait(timeout)

    def is_set(self,
               event_name: str = None,
               is_keyboard: Optional[bool] = False):
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        return self._events[event_name].is_set()

    def clear(self,
              event_name: str = None,
              is_keyboard: Optional[bool] = False):
        if is_keyboard:
            event_name = self._get_keyboard_event_name(event_name)
        return self._events[event_name].clear()

    @staticmethod
    def _get_keyboard_event_name(key):
        return f'_keyboard_{chr(key) if isinstance(key,int) else key}'

    @contextmanager
    def wait_and_handle(self,
                        event_name: str = None,
                        is_keyboard: Optional[bool] = False):
        self.wait(event_name, is_keyboard)
        try:
            yield
        finally:
            self.clear(event_name, is_keyboard)
