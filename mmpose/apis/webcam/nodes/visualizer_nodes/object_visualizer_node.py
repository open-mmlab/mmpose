# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer
from ...utils import FrameMessage
from ..base_visualizer_node import BaseVisualizerNode
from ..registry import NODES


@NODES.register_module()
class ObjectVisualizerNode(BaseVisualizerNode):
    """Visualize the bounding box and keypoints of objects.

    Args:
        name (str): The node name (also thread name)
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
        show_bbox (bool): Set ``True`` to show the bboxes of detection
            objects. Default: ``True``
        show_keypoint (bool): Set ``True`` to show the pose estimation
            results. Default: ``True``
        must_have_bbox (bool): Only show objects with keypoints.
            Default: ``False``
        kpt_thr (float): The threshold of keypoint score. Default: 0.3
        radius (int): The radius of keypoint. Default: 4
        thickness (int): The thickness of skeleton. Default: 2
        bbox_color (str|tuple|dict): The color of bboxes. If a single color is
            given (a str like 'green' or a BGR tuple like (0, 255, 0)), it
            will be used for all bboxes. If a dict is given, it will be used
            as a map from class labels to bbox colors. If not given, a default
            color map will be used. Default: ``None``

    Example::
        >>> cfg = dict(
        ...    type='ObjectVisualizerNode',
        ...    name='object visualizer',
        ...    enable_key='v',
        ...    enable=True,
        ...    show_bbox=True,
        ...    must_have_keypoint=False,
        ...    show_keypoint=True,
        ...    input_buffer='frame',
        ...    output_buffer='vis')

        >>> from mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    default_bbox_color = {
        'person': (148, 139, 255),
        'cat': (255, 255, 0),
        'dog': (255, 255, 0),
    }

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 show_bbox: bool = True,
                 show_keypoint: bool = True,
                 must_have_keypoint: bool = False,
                 kpt_thr: float = 0.3,
                 radius: int = 4,
                 thickness: int = 2,
                 bbox_color: Optional[Union[str, Tuple, Dict]] = 'green'):

        super().__init__(
            name=name,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            enable_key=enable_key,
            enable=enable)

        self.kpt_thr = kpt_thr
        self.bbox_color = bbox_color
        self.show_bbox = show_bbox
        self.show_keypoint = show_keypoint
        self.must_have_keypoint = must_have_keypoint

        self.visualizer = PoseLocalVisualizer(
            name='webcam', radius=radius, line_width=thickness)

    def draw(self, input_msg: FrameMessage) -> np.ndarray:
        canvas = input_msg.get_image()

        if self.must_have_keypoint:
            objects = input_msg.get_objects(
                lambda x: 'bbox' in x and 'keypoints' in x)
        else:
            objects = input_msg.get_objects(lambda x: 'bbox' in x)
        # return if there is no detected objects
        if not objects:
            return canvas

        objects_by_label = defaultdict(list)
        for object in objects:
            objects_by_label[object['label']].append(object)

        # draw objects of each category individually
        for label, objects in objects_by_label.items():
            dataset_meta = objects[0]['dataset_meta']
            dataset_meta['bbox_color'] = self.default_bbox_color.get(
                label, self.bbox_color)
            self.visualizer.set_dataset_meta(dataset_meta)

            # assign bboxes, keypoints and other predictions to data_sample
            instances = InstanceData()
            instances.bboxes = np.stack([object['bbox'] for object in objects])
            instances.labels = np.array(
                [object['class_id'] for object in objects])
            if self.show_keypoint:
                keypoints = [
                    object['keypoints'] for object in objects
                    if 'keypoints' in object
                ]
                if len(keypoints):
                    instances.keypoints = np.stack(keypoints)
                keypoint_scores = [
                    object['keypoint_scores'] for object in objects
                    if 'keypoint_scores' in object
                ]
                if len(keypoint_scores):
                    instances.keypoint_scores = np.stack(keypoint_scores)
            data_sample = PoseDataSample()
            data_sample.pred_instances = instances

            self.visualizer.add_datasample(
                'result',
                canvas,
                data_sample=data_sample,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show=False,
                wait_time=0,
                out_file=None,
                kpt_score_thr=self.kpt_thr)
            canvas = self.visualizer.get_image()

        return canvas
