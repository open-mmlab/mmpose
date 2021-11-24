# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from ..webcam_utils import Message
from .builder import NODES
from .node import Node

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


@NODES.register_module()
class MMDetModelNode(Node):

    def __init__(self,
                 model_config: str,
                 model_checkpoint: str,
                 device: str,
                 name: Optional[str] = None,
                 enable_key=None,
                 input_buffer: str = 'input',
                 output_buffer: str = 'output'):
        # Check mmdetection is installed
        assert has_mmdet, 'Please install mmdet to run the demo.'
        super().__init__(name=name, enable_key=enable_key)

        # Init model
        self.model = init_detector(
            model_config, model_checkpoint, device=device.lower())

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', essential=True)
        self.register_output_buffer(output_buffer)

    def process(self, input_msgs):
        input_msg = input_msgs['input']

        img = input_msg.data['img']

        preds = inference_detector(self.model, img)
        det_result = self._post_process(preds)

        output_data = input_msg.data.copy()
        if 'detection_results' not in output_data:
            output_data['detection_results'] = []

        output_data['detection_results'].append(det_result)

        return Message(data=output_data)

    def _post_process(self, preds):
        if isinstance(preds, tuple):
            preds = preds[0]

        assert len(preds) == len(self.model.CLASSES)
        results = {'results': [], 'model_cfg': self.model.cfg.copy()}

        for i, (cls_name, bboxes) in enumerate(zip(self.model.CLASSES, preds)):
            _results_i = [{
                'cls_id': i,
                'cls_name': cls_name,
                'bbox': bbox
            } for bbox in bboxes]
            results['results'].append(_results_i)

        return results
