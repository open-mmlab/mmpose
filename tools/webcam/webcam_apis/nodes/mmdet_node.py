# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np

from .builder import NODES
from .node import MultiInputNode, Node

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


@NODES.register_module()
class DetectorNode(Node):

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 device: str = 'cuda:0'):
        # Check mmdetection is installed
        assert has_mmdet, \
            f'MMDetection is required for {self.__class__.__name__}.'

        super().__init__(name=name, enable_key=enable_key, enable=enable)

        self.model_config = model_config
        self.model_checkpoint = model_checkpoint
        self.device = device.lower()

        # Init model
        self.model = init_detector(
            self.model_config, self.model_checkpoint, device=self.device)

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def bypass(self, input_msgs):
        return input_msgs['input']

    def process(self, input_msgs):
        input_msg = input_msgs['input']

        img = input_msg.get_image()

        preds = inference_detector(self.model, img)
        det_result = self._post_process(preds)

        input_msg.add_detection_result(det_result, tag=self.name)
        return input_msg

    def _post_process(self, preds):
        if isinstance(preds, tuple):
            dets = preds[0]
            segms = preds[1]
        else:
            dets = preds
            segms = [None] * len(dets)

        det_model_classes = self.model.CLASSES
        if isinstance(det_model_classes, str):
            det_model_classes = (det_model_classes, )

        assert len(dets) == len(det_model_classes)
        assert len(segms) == len(det_model_classes)
        result = {'preds': [], 'model_cfg': self.model.cfg.copy()}

        for i, (cls_name, bboxes,
                masks) in enumerate(zip(det_model_classes, dets, segms)):
            if masks is None:
                masks = [None] * len(bboxes)
            else:
                assert len(masks) == len(bboxes)

            preds_i = [{
                'cls_id': i,
                'label': cls_name,
                'bbox': bbox,
                'mask': mask,
            } for (bbox, mask) in zip(bboxes, masks)]
            result['preds'].extend(preds_i)

        return result


@NODES.register_module()
class MultiFrameDetectorNode(DetectorNode, MultiInputNode):

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 inference_frame: str = 'mid',
                 enable_key: Optional[Union[str, int]] = None,
                 device: str = 'cuda:0'):
        super(MultiFrameDetectorNode,
              self).__init__(name, model_config, model_checkpoint,
                             input_buffer, output_buffer, enable_key, device)
        self.inference_frame = inference_frame

    def process(self, input_msgs):
        input_msg = input_msgs['input']
        if self.inference_frame == 'last':
            key_frame = input_msg[-1]
        elif self.inference_frame == 'mid':
            key_frame = input_msg[len(input_msg) // 2]
        elif self.inference_frame == 'begin':
            key_frame = input_msg[0]
        else:
            raise ValueError(f'Invalid inference_frame {self.inference_frame}')

        img = key_frame.get_image()

        preds = inference_detector(self.model, img)
        det_result = self._post_process(preds)

        imgs = [frame.get_image() for frame in input_msg]
        key_frame.set_image(np.stack(imgs, axis=0))

        key_frame.add_detection_result(det_result, tag=self.name)
        return key_frame
