# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from threading import Event

from mmpose.apis.webcam.utils.event import EventManager


class TestEventManager(unittest.TestCase):

    def test_event_manager(self):
        event_manager = EventManager()

        # test register_event
        event_manager.register_event('example_event')
        self.assertIn('example_event', event_manager._events)
        self.assertIsInstance(event_manager._events['example_event'], Event)
        self.assertFalse(event_manager.is_set('example_event'))

        # test event operations
        event_manager.set('q', is_keyboard=True)
        self.assertIn('_keyboard_q', event_manager._events)
        self.assertTrue(event_manager.is_set('q', is_keyboard=True))

        flag = event_manager.wait('q', is_keyboard=True)
        self.assertTrue(flag)

        event_manager.wait_and_handle('q', is_keyboard=True)
        event_manager.clear('q', is_keyboard=True)
        self.assertFalse(event_manager._events['_keyboard_q']._flag)


if __name__ == '__main__':
    unittest.main()
