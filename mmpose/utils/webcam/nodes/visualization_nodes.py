# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmpose.apis import vis_pose_result
from ..webcam_utils import Message
from .builder import NODES
from .node import Node


@NODES.register_module()
class PoseVisualizerNode(Node):

    def __init__(self,
                 name: str,
                 enable_key=None,
                 kpt_thr: float = 0.3,
                 radius: int = 4,
                 thickness: int = 2,
                 bbox_color: Union[str, tuple] = 'green',
                 frame_buffer: str = '_frame_',
                 result_buffer: Optional[str] = None,
                 output_buffer: str = '_output_'):
        super().__init__(name=name, enable_key=enable_key)

        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness
        self.bbox_color = bbox_color

        # Register buffers
        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        if result_buffer is None:
            self.show_result = False
        else:
            self.register_input_buffer(result_buffer, 'result')
            self.show_result = True
        self.register_output_buffer(output_buffer)

    def _show_result(self, frame_msg, result_msg):

        if 'pose_result_list' not in result_msg.data:
            return Message(data=frame_msg.data)

        img = frame_msg.data['img']
        for pose_result in result_msg.data['pose_result_list']:
            model = pose_result['model_ref']()
            preds = pose_result['preds']
            img = vis_pose_result(
                model,
                img,
                result=preds,
                radius=self.radius,
                thickness=self.thickness,
                kpt_score_thr=self.kpt_thr,
                bbox_color=self.bbox_color)

        return Message(data=dict(img=img))

    def process(self, input_msgs):
        frame_msg = input_msgs['frame']

        if self.show_result:
            result_msg = input_msgs['result']
            output_msg = self.show_result(frame_msg, result_msg)

        else:
            output_msg = Message(data=frame_msg.data)

        return output_msg
