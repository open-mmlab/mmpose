# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from copy import deepcopy

import mmcv
import numpy as np

from mmpose.apis.webcam.nodes import TopdownPoseEstimatorNode
from mmpose.apis.webcam.utils.message import FrameMessage


class TestTopdownPoseEstimatorNode(unittest.TestCase):
    model_config = dict(
        name='human pose estimator',
        model_config='configs/wholebody_2d_keypoint/'
        'topdown_heatmap/coco-wholebody/'
        'td-hm_vipnas-mbv3_dark-8xb64-210e_coco-wholebody-256x192.py',
        model_checkpoint='https://download.openmmlab.com/mmpose/'
        'top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
        '-e2158108_20211205.pth',
        device='cpu',
        input_buffer='det_result',
        output_buffer='human_pose')

    def _get_input_msg(self):

        msg = FrameMessage(None)

        image_path = 'tests/data/coco/000000000785.jpg'
        image = mmcv.imread(image_path)
        msg.set_image(image)

        objects = [
            dict(
                label='person',
                class_id=0,
                bbox=np.array([285.1, 44.4, 510.2, 387.7]))
        ]
        msg.update_objects(objects)

        return msg

    def test_init(self):
        node = TopdownPoseEstimatorNode(**self.model_config)

        self.assertEqual(len(node._input_buffers), 1)
        self.assertEqual(len(node._output_buffers), 1)
        self.assertEqual(node._input_buffers[0].buffer_name, 'det_result')
        self.assertEqual(node._output_buffers[0].buffer_name, 'human_pose')
        self.assertEqual(node.device, 'cpu')

    def test_process(self):
        node = TopdownPoseEstimatorNode(**self.model_config)

        input_msg = self._get_input_msg()
        self.assertEqual(len(input_msg.get_objects()), 1)

        # run inference on all objects
        output_msg = node.process(dict(input=input_msg))
        objects = output_msg.get_objects()

        # there is a person in the image
        self.assertGreaterEqual(len(objects), 1)
        self.assertIn('person', [obj['label'] for obj in objects])
        self.assertEqual(objects[0]['keypoints'].shape, (133, 2))
        self.assertEqual(objects[0]['keypoint_scores'].shape, (133, ))

        # select objects by class_id
        model_config = self.model_config.copy()
        model_config['class_ids'] = [0]
        node = TopdownPoseEstimatorNode(**model_config)
        output_msg = node.process(dict(input=input_msg))
        self.assertGreaterEqual(len(objects), 1)

        # select objects by label
        model_config = self.model_config.copy()
        model_config['labels'] = ['cat']
        node = TopdownPoseEstimatorNode(**model_config)
        output_msg = node.process(dict(input=input_msg))
        self.assertGreaterEqual(len(objects), 0)

    def test_bypass(self):
        node = TopdownPoseEstimatorNode(**self.model_config)

        input_msg = self._get_input_msg()
        input_objects = input_msg.get_objects()

        output_msg = node.bypass(dict(input=deepcopy(input_msg)))
        output_objects = output_msg.get_objects()
        self.assertEqual(len(input_objects), len(output_objects))
        self.assertListEqual(
            list(input_objects[0].keys()), list(output_objects[0].keys()))


if __name__ == '__main__':
    unittest.main()
