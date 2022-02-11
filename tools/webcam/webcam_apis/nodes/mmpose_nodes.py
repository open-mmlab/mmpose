# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Dict, List, Optional, Union

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model)
from ..utils import Message
from .builder import NODES
from .node import Node


@NODES.register_module()
class TopDownPoseEstimatorNode(Node):

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 device: str = 'cuda:0',
                 cls_ids: Optional[List] = None,
                 cls_names: Optional[List] = None,
                 bbox_thr: float = 0.5):
        super().__init__(name=name, enable_key=enable_key, enable=enable)

        # Init model
        self.model_config = model_config
        self.model_checkpoint = model_checkpoint
        self.device = device.lower()

        self.cls_ids = cls_ids
        self.cls_names = cls_names
        self.bbox_thr = bbox_thr

        # Init model
        self.model = init_pose_model(
            self.model_config,
            self.model_checkpoint,
            device=self.device.lower())

        # Store history for pose tracking
        self.track_info = {
            'next_id': 0,
            'last_pose_preds': [],
            'last_time': None
        }

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', essential=True)
        self.register_output_buffer(output_buffer)

    def bypass(self, input_msgs):
        return input_msgs['input']

    def process(self, input_msgs: Dict[str, Message]) -> Message:

        input_msg = input_msgs['input']
        img = input_msg.get_image()
        det_results = input_msg.get_detection_results()

        if det_results is None:
            raise ValueError(
                'No detection results are found in the frame message.'
                f'{self.__class__.__name__} should be used after a '
                'detector node.')

        full_det_preds = []
        for det_result in det_results:
            det_preds = det_result['preds']
            if self.cls_ids:
                # Filter detection results by class ID
                det_preds = [
                    p for p in det_preds if p['cls_id'] in self.cls_ids
                ]
            elif self.cls_names:
                # Filter detection results by class name
                det_preds = [
                    p for p in det_preds if p['label'] in self.cls_names
                ]
            full_det_preds.extend(det_preds)

        # Inference pose
        pose_preds, _ = inference_top_down_pose_model(
            self.model,
            img,
            full_det_preds,
            bbox_thr=self.bbox_thr,
            format='xyxy')

        # Pose tracking
        current_time = time.time()
        if self.track_info['last_time'] is None:
            fps = None
        elif self.track_info['last_time'] >= current_time:
            fps = None
        else:
            fps = 1.0 / (current_time - self.track_info['last_time'])

        pose_preds, next_id = get_track_id(
            pose_preds,
            self.track_info['last_pose_preds'],
            self.track_info['next_id'],
            use_oks=False,
            tracking_thr=0.3,
            use_one_euro=True,
            fps=fps)

        self.track_info['next_id'] = next_id
        self.track_info['last_pose_preds'] = pose_preds.copy()
        self.track_info['last_time'] = current_time

        pose_result = {
            'preds': pose_preds,
            'model_cfg': self.model.cfg.copy(),
        }

        input_msg.add_pose_result(pose_result, tag=self.name)

        return input_msg
