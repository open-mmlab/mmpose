# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmpose.apis import vis_pose_result
from ..webcam_utils import FrameMessage
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

        self.last_results = None

        # Register buffers
        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        if result_buffer is None:
            self.show_result = False
        else:
            self.register_input_buffer(result_buffer, 'result')
            self.show_result = True
        self.register_output_buffer(output_buffer)

    def _show_results(self, img, results):

        for pose_result in results:
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

        return img

    def process(self, input_msgs):
        frame_msg = input_msgs['frame']
        img = frame_msg.get_image()

        result_msg = input_msgs['result']

        if result_msg is not None:
            self.last_results = result_msg.get_pose_result()

        if self.last_results:
            img = self._show_results(img, self.last_results)

        return FrameMessage(img)
