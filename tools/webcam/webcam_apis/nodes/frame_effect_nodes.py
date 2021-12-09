# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Optional, Union

import cv2
import numpy as np
from mmcv import Config, color_val

from mmpose.apis import vis_pose_result
from mmpose.core import apply_bugeye_effect, apply_sunglasses_effect
from mmpose.datasets import DatasetInfo
from ..utils import FrameMessage, Message
from .builder import NODES
from .node import Node

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


def _get_eye_keypoint_ids(model_cfg: Config) -> tuple[int, int]:
    """A helpfer function to get the keypoint indices of left and right eyes
    from the model config.

    Args:
        model_cfg (Config): pose model config.

    Returns:
        int: left eye keypoint index.
        int: right eye keypoint index.
    """
    left_eye_idx = None
    right_eye_idx = None

    # try obtaining eye point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        left_eye_idx = dataset_info.keypoint_name2id.get('left_eye', None)
        right_eye_idx = dataset_info.keypoint_name2id.get('right_eye', None)
    except AttributeError:
        left_eye_idx = None
        right_eye_idx = None

    if left_eye_idx is None or right_eye_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {
                'TopDownCocoDataset', 'TopDownCocoWholeBodyDataset'
        }:
            left_eye_idx = 1
            left_eye_idx = 2
        elif dataset_name in {'AnimalPoseDataset', 'AnimalAP10KDataset'}:
            left_eye_idx = 0
            left_eye_idx = 1
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_eye_idx, right_eye_idx


class BaseFrameEffectNodes(Node):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: str,
                 enable_key: Optional[Union[str, int]] = None):

        super().__init__(name=name, enable_key=enable_key)

        # Register buffers
        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        self.register_output_buffer(output_buffer)

    def process(self, input_msgs: dict[str, Message]) -> Union[Message, None]:
        frame_msg = input_msgs['frame']

        # Video ending signal
        if frame_msg is None:
            return frame_msg

        # Draw
        img = self.draw(input_msgs)
        frame_msg.set_image(img)

        return frame_msg

    def bypass(self, input_msgs: dict[str, Message]) -> Union[Message, None]:
        return input_msgs['frame']

    @abstractmethod
    def draw(self, frame_msg: FrameMessage) -> np.ndarray:
        ...


@NODES.register_module()
class PoseVisualizerNode(BaseFrameEffectNodes):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: str,
                 enable_key: Optional[Union[str, int]] = None,
                 kpt_thr: float = 0.3,
                 radius: int = 4,
                 thickness: int = 2,
                 bbox_color: Union[str, tuple] = 'green'):

        super().__init__(name, frame_buffer, output_buffer, enable_key)

        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness
        self.bbox_color = color_val(bbox_color)

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        for pose_result in frame_msg.get_pose_results():
            model = pose_result['model_ref']()
            preds = pose_result['preds']
            canvas = vis_pose_result(
                model,
                canvas,
                result=preds,
                radius=self.radius,
                thickness=self.thickness,
                kpt_score_thr=self.kpt_thr,
                bbox_color=self.bbox_color)

        return canvas


@NODES.register_module()
class SunglassesNode(BaseFrameEffectNodes):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: str,
                 enable_key: Optional[Union[str, int]] = None,
                 src_img_path: Optional[str] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key)

        if src_img_path is None:
            # The image attributes to:
            # https://www.vecteezy.com/free-vector/glass
            # Glass Vectors by Vecteezy
            src_img_path = 'demo/resources/sunglasses.jpg'
        self.src_img = cv2.imread(src_img_path)

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        for pose_result in frame_msg.get_pose_results():
            model = pose_result['model_ref']()
            preds = pose_result['preds']
            left_eye_idx, right_eye_idx = self._get_eye_keypoint_ids(model.cfg)

            canvas = apply_sunglasses_effect(canvas, preds, self.src_img,
                                             left_eye_idx, right_eye_idx)
        return canvas


@NODES.register_module()
class BugEyeNode(BaseFrameEffectNodes):

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        for pose_result in frame_msg.get_pose_results():
            model = pose_result['model_ref']()
            preds = pose_result['preds']
            left_eye_idx, right_eye_idx = self._get_eye_keypoint_ids(model.cfg)

            canvas = apply_bugeye_effect(canvas, preds, self.src_img,
                                         left_eye_idx, right_eye_idx)
        return canvas
