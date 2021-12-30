# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, List, Optional, Union

from ..utils import Message
from .async_node import AsyncNode
from .builder import NODES

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


@NODES.register_module()
class AsyncCPUDetectorNode(AsyncNode):

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 num_workers: Optional[int] = None,
                 worker_timeout: Optional[float] = None,
                 backend: str = 'thread'):
        # Check mmdetection is installed
        assert has_mmdet, 'Please install mmdet to run the demo.'
        super().__init__(
            name=name,
            enable_key=enable_key,
            enable=enable,
            num_workers=num_workers,
            worker_timeout=worker_timeout,
            backend=backend)

        self.model_config = model_config
        self.model_checkpoint = model_checkpoint

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', essential=True)
        self.register_output_buffer(output_buffer)

    def get_pool_initializer(self):
        return _init_global_model

    def get_pool_initargs(self):
        return self.model_config, self.model_checkpoint

    def get_pool_process_func(self):

        return partial(
            _inference_detector, input_name='input', result_tag=self.name)

    def get_pool_bypass_func(self):
        return lambda msgs: msgs['input']


def _init_global_model(config, checkpoint):
    """A helper function to init a global detector.

    This function is for initializing a worker in a multiprocessing.pool.Pool.
    """
    global model
    model = init_detector(config, checkpoint, device='cpu')


def _inference_detector(input_msgs: Dict[str, Message],
                        input_name: str,
                        result_tag: Optional[str] = None):
    """A helper function to perform object detection.

    This function is for inference a global detector on an input frame image in
    a multiprocessing.pool.Pool.
    """

    global model

    input_msg = input_msgs[input_name]
    img = input_msg.get_image()

    preds = inference_detector(model, img)

    # Postprocess detection result
    if isinstance(preds, tuple):
        preds = preds[0]
    assert len(preds) == len(model.CLASSES)

    det_result = {'preds': [], 'model_cfg': model.cfg.copy()}
    for i, (cls_name, bboxes) in enumerate(zip(model.CLASSES, preds)):
        preds_i = [{
            'cls_id': i,
            'label': cls_name,
            'bbox': bbox
        } for bbox in bboxes]
        det_result['preds'].extend(preds_i)

    input_msg.add_detection_result(det_result, tag=result_tag)
    return input_msg
