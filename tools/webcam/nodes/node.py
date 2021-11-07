# Copyright (c) OpenMMLab. All rights reserved.
import time
from abc import ABCMeta, abstractmethod
from threading import Thread


class Node(Thread, metaclass=ABCMeta):

    def __init__(self, name=None, buffers=[], hotkeys=[], enable_key=None):
        super().__init__(name=name, daemon=True)

        self._buffers = buffers
        self._hotkeys = hotkeys
        self._runner = None
        self._enabled = True
        self._enable_key = enable_key

    @property
    def buffers(self):
        return self._buffers

    @property
    def hotkeys(self):
        return self._hotkeys

    @property
    def runner(self):
        return self._runner

    @property
    def enabled(self):
        return self._enabled

    def check_buffer_ready(self):
        for buffer_name in self.buffers:
            if not self.runner.buffer_manager.get_len(buffer_name):
                return False
        return True

    def set_runner(self, runner):
        if self._runner is not None:
            raise RuntimeError(
                f'Node {self.name} has been registered to a runner. '
                'Re-registering is not allowed.')
        else:
            self._runner = runner

    def run(self):
        # toggle enable
        if self.runner.event_manager.is_set_keyboard(self._enable_key):
            self._enabled = not self._enabled
            self.runner.event_manager.clear_keyboard(self._enable_key)

        # process
        if self._enabled and self.check_buffer_ready():
            self.process()
        else:
            time.sleep(0.001)

    @abstractmethod
    def process(self):
        pass
