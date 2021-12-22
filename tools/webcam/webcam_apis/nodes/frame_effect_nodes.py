# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv import Config, color_val

from mmpose.core import (apply_bugeye_effect, apply_sunglasses_effect,
                         imshow_bboxes, imshow_keypoints)
from mmpose.datasets import DatasetInfo
from ..utils import FrameMessage, Message
from .builder import NODES
from .node import Node

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


def _get_eye_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
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


class BaseFrameEffectNode(Node):
    """Base class for Node that draw on single frame images.

    Args:
        name (str, optional): The node name (also thread name).
        frame_buffer (str): The name of the input buffer.
        output_buffer (str|list): The name(s) of the output buffer(s).
        enable_key (str|int, optional): Set a hot-key to toggle enable/disable
            of the node. If an int value is given, it will be treated as an
            ascii code of a key. Please note:
                1. If enable_key is set, the bypass method need to be
                    overridden to define the node behavior when disabled
                2. Some hot-key has been use for particular use. For example:
                    'q', 'Q' and 27 are used for quit
            Default: None
        enable (bool): Default enable/disable status. Default: True.
    """

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True):

        super().__init__(name=name, enable_key=enable_key)

        # Register buffers
        self.register_input_buffer(frame_buffer, 'frame', essential=True)
        self.register_output_buffer(output_buffer)

        self._enabled = enable

    def process(self, input_msgs: Dict[str, Message]) -> Union[Message, None]:
        frame_msg = input_msgs['frame']

        img = self.draw(frame_msg)
        frame_msg.set_image(img)

        return frame_msg

    def bypass(self, input_msgs: Dict[str, Message]) -> Union[Message, None]:
        return input_msgs['frame']

    @abstractmethod
    def draw(self, frame_msg: FrameMessage) -> np.ndarray:
        """Draw on the frame image with information from the single frame.

        Args:
            frame_meg (FrameMessage): The frame to get information from and
                draw on.

        Returns:
            array: The output image
        """


@NODES.register_module()
class PoseVisualizerNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 kpt_thr: float = 0.3,
                 radius: int = 4,
                 thickness: int = 2,
                 bbox_color: Union[str, Tuple] = 'green'):

        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness
        self.bbox_color = color_val(bbox_color)

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()

        if not pose_results:
            return canvas

        for pose_result in frame_msg.get_pose_results():
            model_cfg = pose_result['model_cfg']
            dataset_info = DatasetInfo(model_cfg.dataset_info)

            bbox_preds = []
            bbox_labels = []
            pose_preds = []
            for pred in pose_result['preds']:
                if 'bbox' in pred:
                    bbox_preds.append(pred['bbox'])
                    bbox_labels.append(pred.get('label', None))
                pose_preds.append(pred['keypoints'])

            # Draw bboxes
            if bbox_preds:
                bboxes = np.vstack(bbox_preds)

                imshow_bboxes(
                    canvas,
                    bboxes,
                    labels=bbox_labels,
                    colors=self.bbox_color,
                    text_color='white',
                    thickness=self.thickness,
                    font_scale=0.5,
                    show=False)

            # Draw poses
            if pose_preds:
                imshow_keypoints(
                    canvas,
                    pose_preds,
                    skeleton=dataset_info.skeleton,
                    kpt_score_thr=0.3,
                    pose_kpt_color=dataset_info.pose_kpt_color,
                    pose_link_color=dataset_info.pose_link_color,
                    radius=self.radius,
                    thickness=self.thickness)

        return canvas


@NODES.register_module()
class SunglassesNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 src_img_path: Optional[str] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        if src_img_path is None:
            # The image attributes to:
            # https://www.vecteezy.com/free-vector/glass
            # Glass Vectors by Vecteezy
            src_img_path = 'demo/resources/sunglasses.jpg'
        self.src_img = cv2.imread(src_img_path)

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            left_eye_idx, right_eye_idx = _get_eye_keypoint_ids(model_cfg)

            canvas = apply_sunglasses_effect(canvas, preds, self.src_img,
                                             left_eye_idx, right_eye_idx)
        return canvas


@NODES.register_module()
class BugEyeNode(BaseFrameEffectNode):

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            left_eye_idx, right_eye_idx = _get_eye_keypoint_ids(model_cfg)

            canvas = apply_bugeye_effect(canvas, preds, left_eye_idx,
                                         right_eye_idx)
        return canvas


@NODES.register_module()
class BillboardNode(BaseFrameEffectNode):

    default_content_lines = ['This is a billboard!']

    def __init__(
        self,
        name: str,
        frame_buffer: str,
        output_buffer: Union[str, List[str]],
        enable_key: Optional[Union[str, int]] = None,
        enable: bool = True,
        content_lines: Optional[List[str]] = None,
        x_offset: int = 20,
        y_offset: int = 20,
        y_delta: int = 15,
        text_color: Union[str, Tuple[int, int, int]] = 'black',
        background_color: Union[str, Tuple[int, int, int]] = (255, 183, 0),
        text_scale: float = 0.4,
    ):
        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        self._enabled = True

        self.x_offset = x_offset
        self.y_offset = y_offset
        self.y_delta = y_delta
        self.text_color = color_val(text_color)
        self.background_color = color_val(background_color)
        self.text_scale = text_scale

        if content_lines:
            self.content_lines = content_lines
        else:
            self.content_lines = self.default_content_lines

    def draw(self, frame_msg: FrameMessage) -> np.ndarray:
        img = frame_msg.get_image()
        canvas = np.full(img.shape, self.background_color, dtype=img.dtype)

        x = self.x_offset
        y = self.y_offset

        max_len = max([len(line) for line in self.content_lines])

        def _put_line(line=''):
            nonlocal y
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        self.text_scale, self.text_color, 1)
            y += self.y_delta

        for line in self.content_lines:
            _put_line(line)

        x1 = max(0, self.x_offset)
        x2 = min(img.shape[1], int(x + max_len * self.text_scale * 20))
        y1 = max(0, self.y_offset - self.y_delta)
        y2 = min(img.shape[0], y)

        src1 = canvas[y1:y2, x1:x2]
        src2 = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

        return img
