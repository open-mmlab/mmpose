# Copyright (c) OpenMMLab. All rights reserved.
import weakref
from typing import Optional

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model)
from ..webcam_utils import Message
from .builder import NODES
from .node import Node


@NODES.register_module()
class TopDownPoseEstimatorNode(Node):

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 device: str,
                 cls_ids: Optional[list] = None,
                 bbox_thr: float = 0.5,
                 enable_key=None,
                 input_buffer: str = 'input',
                 output_buffer: str = 'output'):
        super().__init__(name=name, enable_key=enable_key)

        # Init model
        self.model = init_pose_model(
            model_config, model_checkpoint, device=device.lower())

        self.cls_ids = cls_ids
        self.bbox_thr = bbox_thr

        # Store history for pose tracking
        self.track_info = {'next_id': 0, 'last_pose_preds': []}

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', essential=True)
        self.register_output_buffer(output_buffer)

    def process(self, input_msgs: dict[str, Message]) -> Message:

        input_msg = input_msgs['input']
        img = input_msg.data['img']
        det_result = input_msg.data['detection_result']

        det_preds = det_result['preds']
        if self.cls_ids:
            det_preds = [
                pred for pred in det_preds if pred['cls_id'] in self.cls_ids
            ]

        # Inference pose
        pose_preds, _ = inference_top_down_pose_model(
            self.model, img, det_preds, bbox_thr=self.bbox_thr, format='xyxy')

        # Pose tracking
        pose_preds, next_id = get_track_id(
            pose_preds,
            self.track_info['last_pose_preds'],
            self.track_info['next_id'],
            use_oks=False,
            tracking_thr=0.3,
            use_one_euro=True,
            fps=None)

        self.track_info['next_id'] = next_id
        self.track_info['last_pose_preds'] = pose_preds.copy()

        pose_result = {
            'preds': pose_preds,
            'model_ref': weakref.ref(self.model)
        }

        output_data = input_msg.data.copy()

        if 'pose_result_list' not in output_data:
            output_data['pose_result_list'] = []
        output_data['pose_result_list'].append(pose_result)

        return Message(data=output_data)
