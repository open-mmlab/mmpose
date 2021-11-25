# Copyright (c) OpenMMLab. All rights reserved.
import weakref

from ..webcam_utils import Message
from .builder import NODES
from .node import Node

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
                 device: str,
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
        output_data['detection_result'] = det_result

        return Message(data=output_data)

    def _post_process(self, preds):
        if isinstance(preds, tuple):
            preds = preds[0]

        assert len(preds) == len(self.model.CLASSES)
        result = {'preds': [], 'model_ref': weakref.ref(self.model)}

        for i, (cls_name, bboxes) in enumerate(zip(self.model.CLASSES, preds)):
            _preds_i = [{
                'cls_id': i,
                'cls_name': cls_name,
                'bbox': bbox
            } for bbox in bboxes]
            result['preds'].append(_preds_i)

        return result
