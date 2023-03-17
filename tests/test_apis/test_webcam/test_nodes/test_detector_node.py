# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import mmcv

from mmpose.apis.webcam.nodes import DetectorNode
from mmpose.apis.webcam.utils.message import FrameMessage


class TestDetectorNode(unittest.TestCase):
    model_config = dict(
        name='detector',
        model_config='demo/mmdetection_cfg/'
        'ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py',
        model_checkpoint='https://download.openmmlab.com'
        '/mmdetection/v2.0/ssd/'
        'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
        'scratch_600e_coco_20210629_110627-974d9307.pth',
        device='cpu',
        input_buffer='_input_',
        output_buffer='det_result')

    def setUp(self) -> None:
        self._has_mmdet = True
        try:
            from mmdet.apis import init_detector  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            self._has_mmdet = False

    def _get_input_msg(self):

        msg = FrameMessage(None)

        image_path = 'tests/data/coco/000000000785.jpg'
        image = mmcv.imread(image_path)
        msg.set_image(image)

        return msg

    def test_init(self):

        if not self._has_mmdet:
            return unittest.skip('mmdet is not installed')

        node = DetectorNode(**self.model_config)

        self.assertEqual(len(node._input_buffers), 1)
        self.assertEqual(len(node._output_buffers), 1)
        self.assertEqual(node._input_buffers[0].buffer_name, '_input_')
        self.assertEqual(node._output_buffers[0].buffer_name, 'det_result')
        self.assertEqual(node.device, 'cpu')

    def test_process(self):

        if not self._has_mmdet:
            return unittest.skip('mmdet is not installed')

        node = DetectorNode(**self.model_config)

        input_msg = self._get_input_msg()
        self.assertEqual(len(input_msg.get_objects()), 0)

        output_msg = node.process(dict(input=input_msg))
        objects = output_msg.get_objects()
        # there is a person in the image
        self.assertGreaterEqual(len(objects), 1)
        self.assertIn('person', [obj['label'] for obj in objects])
        self.assertEqual(objects[0]['bbox'].shape, (4, ))

    def test_bypass(self):

        if not self._has_mmdet:
            return unittest.skip('mmdet is not installed')

        node = DetectorNode(**self.model_config)

        input_msg = self._get_input_msg()
        self.assertEqual(len(input_msg.get_objects()), 0)

        output_msg = node.bypass(dict(input=input_msg))
        self.assertEqual(len(output_msg.get_objects()), 0)


if __name__ == '__main__':
    unittest.main()
