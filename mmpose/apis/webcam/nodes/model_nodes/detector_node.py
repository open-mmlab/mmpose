# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np

from ...utils import get_config_path
from ..node import Node
from ..registry import NODES

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


@NODES.register_module()
class DetectorNode(Node):
    """Detect objects from the frame image using MMDetection model.

    Note that MMDetection is required for this node. Please refer to
    `MMDetection documentation <https://mmdetection.readthedocs.io/en
    /latest/get_started.html>`_ for the installation guide.

    Parameters:
        name (str): The node name (also thread name)
        model_cfg (str): The model config file
        model_checkpoint (str): The model checkpoint file
        input_buffer (str): The name of the input buffer
        output_buffer (str|list): The name(s) of the output buffer(s)
        enable_key (str|int, optional): Set a hot-key to toggle enable/disable
            of the node. If an int value is given, it will be treated as an
            ascii code of a key. Please note: (1) If ``enable_key`` is set,
            the ``bypass()`` method need to be overridden to define the node
            behavior when disabled; (2) Some hot-keys are reserved for
            particular use. For example: 'q', 'Q' and 27 are used for exiting.
            Default: ``None``
        enable (bool): Default enable/disable status. Default: ``True``
        device (str): Specify the device to hold model weights and inference
            the model. Default: ``'cuda:0'``
        bbox_thr (float): Set a threshold to filter out objects with low bbox
            scores. Default: 0.5
        multi_input (bool): Whether load all frames in input buffer. If True,
            all frames in buffer will be loaded and stacked. The latest frame
            is used to detect objects of interest. Default: False

    Example::
        >>> cfg = dict(
        ...     type='DetectorNode',
        ...     name='detector',
        ...     model_config='demo/mmdetection_cfg/'
        ...     'ssdlite_mobilenetv2_scratch_600e_coco.py',
        ...     model_checkpoint='https://download.openmmlab.com'
        ...     '/mmdetection/v2.0/ssd/'
        ...     'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
        ...     'scratch_600e_coco_20210629_110627-974d9307.pth',
        ...     # `_input_` is an executor-reserved buffer
        ...     input_buffer='_input_',
        ...     output_buffer='det_result')

        >>> from mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 device: str = 'cuda:0',
                 bbox_thr: float = 0.5,
                 multi_input: bool = False):
        # Check mmdetection is installed
        assert has_mmdet, \
            f'MMDetection is required for {self.__class__.__name__}.'

        super().__init__(
            name=name,
            enable_key=enable_key,
            enable=enable,
            multi_input=multi_input)

        self.model_config = get_config_path(model_config, 'mmdet')
        self.model_checkpoint = model_checkpoint
        self.device = device.lower()
        self.bbox_thr = bbox_thr

        # Init model
        register_all_modules()
        self.model = init_detector(
            self.model_config, self.model_checkpoint, device=self.device)

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def bypass(self, input_msgs):
        return input_msgs['input']

    def process(self, input_msgs):
        input_msg = input_msgs['input']

        if self.multi_input:
            imgs = [frame.get_image() for frame in input_msg]
            input_msg = input_msg[-1]

        img = input_msg.get_image()

        register_all_modules()
        preds = inference_detector(self.model, img)
        objects = self._post_process(preds)
        input_msg.update_objects(objects)

        if self.multi_input:
            input_msg.set_image(np.stack(imgs, axis=0))

        return input_msg

    def _post_process(self, preds) -> List[Dict]:
        """Post-process the predictions of MMDetection model."""
        instances = preds.pred_instances.cpu().numpy()

        classes = self.model.dataset_meta['CLASSES']
        if isinstance(classes, str):
            classes = (classes, )

        objects = []
        for i in range(len(instances)):
            if instances.scores[i] < self.bbox_thr:
                continue
            class_id = instances.labels[i]
            obj = {
                'class_id': class_id,
                'label': classes[class_id],
                'bbox': instances.bboxes[i],
                'det_model_cfg': self.model.cfg,
                'dataset_meta': self.model.dataset_meta.copy(),
            }
            objects.append(obj)
        return objects
