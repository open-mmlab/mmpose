# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.logging import MessageHub

from mmpose.models.data_preprocessors import (BatchSyncRandomResize,
                                              PoseDataPreprocessor)
from mmpose.structures import PoseDataSample


class TestPoseDataPreprocessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = PoseDataPreprocessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = PoseDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            PoseDataPreprocessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            PoseDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

    def test_forward(self):
        processor = PoseDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = {
            'inputs': [torch.randint(0, 256, (3, 11, 10))],
            'data_samples': [PoseDataSample()]
        }
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']

        self.assertEqual(batch_inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(batch_data_samples), 1)

        # test channel_conversion
        processor = PoseDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(batch_data_samples), 1)

        # test padding
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 10, 11)),
                torch.randint(0, 256, (3, 9, 14))
            ],
            'data_samples': [PoseDataSample()] * 2
        }
        processor = PoseDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs.shape, (2, 3, 10, 14))
        self.assertEqual(len(batch_data_samples), 2)

        # test pad_size_divisor
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 10, 11)),
                torch.randint(0, 256, (3, 9, 24))
            ],
            'data_samples': [PoseDataSample()] * 2
        }
        processor = PoseDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], pad_size_divisor=5)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs.shape, (2, 3, 10, 25))
        self.assertEqual(len(batch_data_samples), 2)
        for data_samples, expected_shape in zip(batch_data_samples,
                                                [(10, 15), (10, 25)]):
            self.assertEqual(data_samples.pad_shape, expected_shape)

    def test_batch_sync_random_resize(self):
        processor = PoseDataPreprocessor(batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(320, 320),
                size_divisor=32,
                interval=1)
        ])
        self.assertTrue(
            isinstance(processor.batch_augments[0], BatchSyncRandomResize))
        message_hub = MessageHub.get_instance('test_batch_sync_random_resize')
        message_hub.update_info('iter', 0)
        packed_inputs = {
            'inputs': [
                torch.randint(0, 256, (3, 128, 128)),
                torch.randint(0, 256, (3, 128, 128))
            ],
            'data_samples': [PoseDataSample()] * 2
        }
        batch_inputs = processor(packed_inputs, training=True)['inputs']
        self.assertEqual(batch_inputs.shape, (2, 3, 128, 128))

        # resize after one iter
        message_hub.update_info('iter', 1)
        packed_inputs = {
            'inputs': [
                torch.randint(0, 256, (3, 128, 128)),
                torch.randint(0, 256, (3, 128, 128))
            ],
            'data_samples':
            [PoseDataSample(metainfo=dict(img_shape=(128, 128)))] * 2
        }
        batch_inputs = processor(packed_inputs, training=True)['inputs']
        self.assertEqual(batch_inputs.shape, (2, 3, 320, 320))

        packed_inputs = {
            'inputs': [
                torch.randint(0, 256, (3, 128, 128)),
                torch.randint(0, 256, (3, 128, 128))
            ],
            'data_samples': [PoseDataSample()] * 2
        }
        batch_inputs = processor(packed_inputs, training=False)['inputs']
        self.assertEqual(batch_inputs.shape, (2, 3, 128, 128))
