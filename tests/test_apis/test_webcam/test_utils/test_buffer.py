# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from queue import Queue

from mmpose.apis.webcam.utils.buffer import Buffer, BufferManager


class TestBuffer(unittest.TestCase):

    def test_buffer(self):

        buffer = Buffer(maxsize=1)
        for i in range(3):
            buffer.put_force(i)
        item = buffer.get()
        self.assertEqual(item, 2)


class TestBufferManager(unittest.TestCase):

    def _get_buffer_dict(self):
        return dict(example_buffer=Buffer())

    def test_init(self):

        # test default initialization
        buffer_manager = BufferManager()
        self.assertIn('_buffers', dir(buffer_manager))
        self.assertIsInstance(buffer_manager._buffers, dict)

        # test initialization with given buffers
        buffers = self._get_buffer_dict()
        buffer_manager = BufferManager(buffers=buffers)
        self.assertIn('_buffers', dir(buffer_manager))
        self.assertIsInstance(buffer_manager._buffers, dict)
        self.assertIn('example_buffer', buffer_manager._buffers.keys())
        # test __contains__
        self.assertIn('example_buffer', buffer_manager)

        # test initialization with incorrect buffers
        buffers['incorrect_buffer'] = Queue()
        with self.assertRaises(ValueError):
            buffer_manager = BufferManager(buffers=buffers)

    def test_buffer_operations(self):
        buffer_manager = BufferManager()

        # test register_buffer
        buffer_manager.register_buffer('example_buffer', 1)
        self.assertIn('example_buffer', buffer_manager)
        self.assertEqual(buffer_manager._buffers['example_buffer'].maxsize, 1)

        # test buffer operations
        buffer_manager.put('example_buffer', 0)
        item = buffer_manager.get('example_buffer')
        self.assertEqual(item, 0)

        buffer_manager.put('example_buffer', 0)
        self.assertTrue(buffer_manager.is_full('example_buffer'))
        buffer_manager.put_force('example_buffer', 1)
        item = buffer_manager.get('example_buffer')
        self.assertEqual(item, 1)
        self.assertTrue(buffer_manager.is_empty('example_buffer'))

        # test get_info
        buffer_info = buffer_manager.get_info()
        self.assertIn('example_buffer', buffer_info)
        self.assertEqual(buffer_info['example_buffer']['size'], 0)
        self.assertEqual(buffer_info['example_buffer']['maxsize'], 1)

        # test get_sub_manager
        buffer_manager = buffer_manager.get_sub_manager(['example_buffer'])
        self.assertIsInstance(buffer_manager, BufferManager)
        self.assertIn('example_buffer', buffer_manager)
        self.assertEqual(buffer_manager._buffers['example_buffer'].maxsize, 1)


if __name__ == '__main__':
    unittest.main()
