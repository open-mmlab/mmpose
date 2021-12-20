# Copyright (c) OpenMMLab. All rights reserved.
import time
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv import Config, color_val

from mmpose.core import (apply_bugeye_effect, apply_firecracker_effect,
                         apply_hat_effect, apply_sunglasses_effect,
                         imshow_bboxes, imshow_keypoints)
from mmpose.datasets import DatasetInfo
from ..utils import (FrameMessage, Message, copy_and_paste, expand_and_clamp,
                     get_cached_file_path, load_image_from_disk_or_url,
                     screen_matting)
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
            right_eye_idx = 2
        elif dataset_name in {'AnimalPoseDataset', 'AnimalAP10KDataset'}:
            left_eye_idx = 0
            right_eye_idx = 1
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_eye_idx, right_eye_idx


def _get_nose_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helpfer function to get the keypoint indices of the nose from the
    model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        int: nose keypoint index.
    """
    nose_idx = None

    # try obtaining nose point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        nose_idx = dataset_info.keypoint_name2id.get('nose', None)
    except AttributeError:
        nose_idx = None

    if nose_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {
                'TopDownCocoDataset', 'TopDownCocoWholeBodyDataset'
        }:
            nose_idx = 0
        elif dataset_name in {'AnimalPoseDataset', 'AnimalAP10KDataset'}:
            nose_idx = 2
        else:
            raise ValueError('Can not determine the nose id of '
                             f'{dataset_name}')

    return nose_idx


def _get_ear_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helpfer function to get the keypoint indices of left and right ears
    from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        int: left ear keypoint index.
        int: right ear keypoint index.
    """

    # try obtaining eye point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)

        left_ear_idx = dataset_info.keypoint_name2id.get('left_ear', None)
        right_ear_idx = dataset_info.keypoint_name2id.get('right_ear', None)

    except AttributeError:
        left_ear_idx = None
        right_ear_idx = None

    if left_ear_idx is None or right_ear_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {
                'TopDownCocoDataset', 'TopDownCocoWholeBodyDataset'
        }:
            left_ear_idx = 3
            right_ear_idx = 4
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_ear_idx, right_ear_idx


def _get_face_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helpfer function to get the keypoint indices of the face from the
    model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        list[int]: face keypoint index.
    """
    face_indices = None

    # try obtaining nose point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        for id in range(68):
            face_indices.append(
                dataset_info.keypoint_name2id.get(f'face_{id}', None))
    except AttributeError:
        face_indices = None

    if face_indices is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {'TopDownCocoWholeBodyDataset'}:
            face_indices = list(range(23, 91))
        else:
            raise ValueError('Can not determine the face id of '
                             f'{dataset_name}')

    return face_indices


def _get_wrist_keypoint_ids(model_cfg: Config) -> Tuple[int, int]:
    """A helpfer function to get the keypoint indices of left and right wrist
    from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        int: left wrist keypoint index.
        int: right wrist keypoint index.
    """

    # try obtaining eye point ids from dataset_info
    try:
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)
        left_wrist_idx = dataset_info.keypoint_name2id.get('left_wrist', None)
        right_wrist_idx = dataset_info.keypoint_name2id.get(
            'right_wrist', None)
    except AttributeError:
        left_wrist_idx = None
        right_wrist_idx = None

    if left_wrist_idx is None or right_wrist_idx is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {
                'TopDownCocoDataset', 'TopDownCocoWholeBodyDataset'
        }:
            left_wrist_idx = 9
            right_wrist_idx = 10
        elif dataset_name == 'AnimalPoseDataset':
            left_wrist_idx = 16
            right_wrist_idx = 17
        elif dataset_name == 'AnimalAP10KDataset':
            left_wrist_idx = 7
            right_wrist_idx = 10
        else:
            raise ValueError('Can not determine the eye keypoint id of '
                             f'{dataset_name}')

    return left_wrist_idx, right_wrist_idx


def _get_hand_keypoint_ids(model_cfg: Config) -> List[int]:
    """A helpfer function to get the keypoint indices of left and right hand
    from the model config.

    Args:
        model_cfg (Config): pose model config.
    Returns:
        list[int]: hand keypoint indices.
    """
    # try obtaining hand keypoint ids from dataset_info
    try:
        hand_indices = []
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)

        hand_indices.append(
            dataset_info.keypoint_name2id.get('left_hand_root', None))

        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_thumb{id}', None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_forefinger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_middle_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_ring_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'left_pinky_finger{id}',
                                                  None))

        hand_indices.append(
            dataset_info.keypoint_name2id.get('right_hand_root', None))

        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_thumb{id}', None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_forefinger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_middle_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_ring_finger{id}',
                                                  None))
        for id in range(1, 5):
            hand_indices.append(
                dataset_info.keypoint_name2id.get(f'right_pinky_finger{id}',
                                                  None))

    except AttributeError:
        hand_indices = None

    if hand_indices is None:
        # Fall back to hard coded keypoint id
        dataset_name = model_cfg.data.test.type
        if dataset_name in {'TopDownCocoWholeBodyDataset'}:
            hand_indices = list(range(91, 133))
        else:
            raise ValueError('Can not determine the hand id of '
                             f'{dataset_name}')

    return hand_indices


def modify_dataset_info(dataset_info: DatasetInfo) -> DatasetInfo:
    # remove skeleton above face
    for i in range(12, 17):
        dataset_info.skeleton_info.pop(i)

    # connect 17 face keypoints
    for i in range(65, 65 + 16):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-65}', f'face-{i-64}'), id=i, color=[255, 255, 255])
    # connect above eyebow keypoints
    for i in range(81, 81 + 4):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-64}', f'face-{i-63}'), id=i, color=[255, 255, 255])
    for i in range(85, 85 + 4):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-63}', f'face-{i-62}'), id=i, color=[255, 255, 255])
    # connect nose keypoints
    for i in range(89, 89 + 3):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-62}', f'face-{i-61}'), id=i, color=[255, 255, 255])
    for i in range(92, 92 + 4):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-61}', f'face-{i-60}'), id=i, color=[255, 255, 255])
    # connect eye keypoints
    for i in range(96, 96 + 5):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-60}', f'face-{i-59}'), id=i, color=[255, 255, 255])
    dataset_info.skeleton_info[101] = dict(
        link=('face-36', 'face-41'), id=101, color=[255, 255, 255])
    for i in range(102, 102 + 5):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-60}', f'face-{i-59}'), id=i, color=[255, 255, 255])
    dataset_info.skeleton_info[107] = dict(
        link=('face-42', 'face-47'), id=107, color=[255, 255, 255])
    # connect mouth keypoints
    for i in range(108, 108 + 11):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-60}', f'face-{i-59}'), id=i, color=[255, 255, 255])
    dataset_info.skeleton_info[119] = dict(
        link=('face-48', 'face-59'), id=119, color=[255, 255, 255])
    for i in range(120, 120 + 7):
        dataset_info.skeleton_info[i] = dict(
            link=(f'face-{i-60}', f'face-{i-59}'), id=i, color=[255, 255, 255])

    return dataset_info


def is_person_visible(pred: Dict[str, np.ndarray], kpts_indices: Tuple[int],
                      kpt_thr: float) -> bool:
    for i in kpts_indices:
        if pred['keypoints'][i][2] < kpt_thr:
            return False

    return True


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
    """Draw the bbox and keypoint detection results.

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
        kpt_thr (float): The threshold of keypoint score. Default: 0.3.
        radius (int): The radius of keypoint. Default: 4.
        thickness (int): The thickness of skeleton. Default: 2.
        bbox_color (str|tuple|dict): If a single color (a str like 'green' or
            a tuple like (0, 255, 0)), it will used to draw the bbox.
            Optionally, a dict can be given as a map from class labels to
            colors.
    """

    default_bbox_color = {
        'person': (148, 139, 255),
        'cat': (255, 255, 0),
        'dog': (255, 255, 0),
    }

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 kpt_thr: float = 0.3,
                 radius: int = 4,
                 thickness: int = 2,
                 bbox_color: Optional[Union[str, Tuple, Dict]] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness
        if bbox_color is None:
            self.bbox_color = self.default_bbox_color
        elif isinstance(bbox_color, dict):
            self.bbox_color = {k: color_val(v) for k, v in bbox_color.items()}
        else:
            self.bbox_color = color_val(bbox_color)

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()

        if not pose_results:
            return canvas

        for pose_result in frame_msg.get_pose_results():
            model_cfg = pose_result['model_cfg']
            dataset_info = DatasetInfo(model_cfg.dataset_info)

            # Extract bboxes and poses
            bbox_preds = []
            bbox_labels = []
            pose_preds = []
            for pred in pose_result['preds']:
                if 'bbox' in pred:
                    bbox_preds.append(pred['bbox'])
                    bbox_labels.append(pred.get('label', None))
                pose_preds.append(pred['keypoints'])

            # Get bbox colors
            if isinstance(self.bbox_color, dict):
                bbox_colors = [
                    self.bbox_color.get(label, (0, 255, 0))
                    for label in bbox_labels
                ]
            else:
                bbox_labels = self.bbox_color

            # Draw bboxes
            if bbox_preds:
                bboxes = np.vstack(bbox_preds)

                imshow_bboxes(
                    canvas,
                    bboxes,
                    labels=bbox_labels,
                    colors=bbox_colors,
                    text_color='white',
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
class MatchStickMenNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 background_color: Union[str, Tuple[int, int,
                                                    int]] = (0, 0, 0),
                 kpt_thr: float = 0.3,
                 radius: int = 2,
                 thickness: int = 2,
                 angle_thr: float = 90.0,
                 heartbeat_duration: float = 2.0,
                 largest_heart: Tuple[int, int] = (256, 256),
                 src_img_path: Optional[str] = None,
                 dis_thr: float = 200.0,
                 essential_kpts_indices: Tuple[int] = (0, 1, 2)):

        super().__init__(
            name, frame_buffer, output_buffer, enable_key=enable_key)

        if src_img_path is None:
            src_img_path = 'https://user-images.githubusercontent.com/'\
                           '87690686/149731850-ea946766-a4e8-4efa-82f5'\
                           '-e2f0515db8ae.png'
        self.src_img = load_image_from_disk_or_url(src_img_path)

        self.background_color = background_color
        self.kpt_thr = kpt_thr
        self.radius = radius
        self.thickness = thickness

        self.angle_thr = angle_thr
        self.dis_thr = dis_thr
        self.heartbeat_duration = heartbeat_duration
        self.largest_heart = largest_heart

        # record the heart_effect start time for each person
        self._heart_start_time = {}
        self._heart_pos = {}
        self.dataset_info = None

        self.essential_kpts_indices = essential_kpts_indices

    def _cal_distance(self, keypoints: np.ndarray,
                      hand_indices: List[str]) -> np.float64:
        # 20, 41
        p1 = keypoints[hand_indices[20]][:2]
        p2 = keypoints[hand_indices[41]][:2]

        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _cal_angle(self, keypoints: np.ndarray, hand_indices: List[int],
                   finger_indices: List[int]) -> np.float64:

        p1 = keypoints[hand_indices[finger_indices[3]]][:2]
        p2 = keypoints[hand_indices[finger_indices[2]]][:2]

        p3 = keypoints[hand_indices[finger_indices[1]]][:2]
        p4 = keypoints[hand_indices[finger_indices[0]]][:2]

        v1 = p2 - p1
        v2 = p4 - p3

        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = np.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * np.sqrt(
            pow(v2[0], 2) + pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)

        return (np.arccos(cos) / np.pi) * 180

    def _check_heart(self, pred: Dict[str, np.ndarray],
                     hand_indices: List[int]) -> bool:
        """Check if a person is posing a 'hand heart' gesture.

        Args:
            pred(dict): The pose estimation results containing:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result
                                                 in [x, y, score]
            hand_indices(list[int]): hand keypoint indices.

        Returns:
            Boolean: whether the person is posing a "hand heart" gesture
        """
        keypoints = pred['keypoints']

        # note: these indices are corresoponding to the following keypoints:
        # left_hand_root, left_pinky_finger1,
        # left_pinky_finger3, left_pinky_finger4,
        # right_hand_root, right_pinky_finger1
        # right_pinky_finger3, right_pinky_finger4
        for i in [0, 17, 19, 20, 21, 38, 40, 41]:
            if keypoints[hand_indices[i]][2] < self.kpt_thr:
                return False

        left_indices = [0, 17, 19, 20]
        left_angle = self._cal_angle(keypoints, hand_indices, left_indices)

        right_indices = [21, 38, 40, 41]
        right_angle = self._cal_angle(keypoints, hand_indices, right_indices)

        dis = self._cal_distance(keypoints, hand_indices)

        if left_angle < self.angle_thr and right_angle < self.angle_thr \
           and dis < self.dis_thr:
            return True

        return False

    def _draw_heart(self, canvas: np.ndarray, heart_pos: Tuple[int, int],
                    t_pass: float) -> np.ndarray:
        scale = t_pass / self.heartbeat_duration
        hm, wm = self.largest_heart
        new_h, new_w = int(hm * scale), int(wm * scale)

        max_h, max_w = canvas.shape[:2]

        x, y = heart_pos
        y1 = max(0, y - int(new_h / 2))
        y2 = min(max_h - 1, y + int(new_h / 2))

        x1 = max(0, x - int(new_w / 2))
        x2 = min(max_w - 1, x + int(new_w / 2))

        target = canvas[y1:y2 + 1, x1:x2 + 1].copy()
        new_h, new_w = target.shape[:2]

        if new_h == 0 or new_w == 0:
            return canvas

        patch = self.src_img.copy()
        patch = cv2.resize(patch, (new_w, new_h))
        mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask = (mask < 100)[..., None].astype(np.float32) * 0.8

        canvas[y1:y2 + 1, x1:x2 + 1] = patch * mask + target * (1 - mask)

        return canvas

    def _get_heart_pos(self, pred: Dict[str, np.ndarray],
                       hand_indices: List[int]) -> Tuple[int, int]:
        keypoints = pred['keypoints']
        p1 = keypoints[hand_indices[20]][:2]
        p2 = keypoints[hand_indices[41]][:2]

        x, y = (p1 + p2) / 2
        # the mid point of two fingers
        return int(x), int(y)

    def draw(self, frame_msg: FrameMessage) -> np.ndarray:
        canvas = frame_msg.get_image()
        canvas[:] = self.background_color

        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas
        for pose_result in frame_msg.get_pose_results():
            pose_preds = []
            preds = [pred.copy() for pred in pose_result['preds']]
            for pred in preds:
                if is_person_visible(pred, self.essential_kpts_indices,
                                     self.kpt_thr):
                    pose_preds.append(pred['keypoints'])

            model_cfg = pose_result['model_cfg']

            if self.dataset_info is None:
                dataset_info = model_cfg.data.test.dataset_info.copy()
                dataset_info = modify_dataset_info(dataset_info)
                self.dataset_info = DatasetInfo(dataset_info)

            if pose_preds:
                imshow_keypoints(
                    canvas,
                    pose_preds,
                    skeleton=self.dataset_info.skeleton,
                    kpt_score_thr=0.3,
                    pose_kpt_color=self.dataset_info.pose_kpt_color,
                    pose_link_color=self.dataset_info.pose_link_color,
                    radius=self.radius,
                    thickness=self.thickness)

            for pred in preds:
                id = pred['track_id']
                if self._heart_start_time.get(id, None) is not None:
                    t_pass = time.time() - self._heart_start_time[id]
                    if t_pass < self.heartbeat_duration:
                        canvas = self._draw_heart(canvas, self._heart_pos[id],
                                                  t_pass)
                    else:
                        self._heart_start_time[id] = None
                        self._heart_pos[id] = None
                else:
                    hand_indices = _get_hand_keypoint_ids(model_cfg)
                    if self._check_heart(pred, hand_indices):
                        self._heart_start_time[id] = time.time()
                        self._heart_pos[id] = self._get_heart_pos(
                            pred, hand_indices)

        return canvas


@NODES.register_module()
class ELkHornNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 src_img_path: Optional[str] = None,
                 kpt_thr: float = 0.3,
                 anchor_points_indices=[23, 39, 29, 33]):

        super().__init__(name, frame_buffer, output_buffer, enable_key)

        if src_img_path is None:
            src_img_path = 'https://user-images.githubusercontent.com/'\
                           '87690686/149731877-1a7ff0f3-fc5a-4fd5-b330'\
                           '-7f35e2930f02.jpg'

        self.src_img = load_image_from_disk_or_url(src_img_path)
        # The score threshold of required keypoints.
        self.kpt_thr = kpt_thr
        self.anchor_points_indices = anchor_points_indices

    def draw(self, frame_msg: FrameMessage) -> np.ndarray:
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas
        for pose_result in pose_results:
            for pred in pose_result['preds']:
                if is_person_visible(pred, self.anchor_points_indices,
                                     self.kpt_thr):
                    canvas = self.apply_elk_horn_effect(canvas, pred)
        return canvas

    def apply_elk_horn_effect(self, canvas: np.ndarray,
                              pose: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply elk_horn effect.

        Args:
            canvas (np.ndarray): Image data.
            pose(dict): The pose estimation results containing:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result
                                                 in [x, y, score]
        """
        # anchor points in the elk horn mask
        pts_src = np.array([[260, 580], [680, 580], [260, 900], [680, 900]],
                           dtype=np.float32)

        for i in self.anchor_points_indices:
            if pose['keypoints'][i][2] < self.kpt_thr:
                continue
        # choose 4 anchor points, the keypoint indices can be found under
        # 'configs/_base_/datasets/coco_wholebody.py'
        # 23: Keypoint index of 'face-0'
        # 39: Keypoint index of 'face-16'
        # 29: Keypoint index of 'face-6'
        # 33: Keypoint index of 'face-10'

        kpt_0 = pose['keypoints'][23][:2]
        kpt_16 = pose['keypoints'][39][:2]
        # orthogonal vector
        # decide whether need to reverse the order
        if kpt_0[0] < kpt_16[0]:
            vo = (kpt_0 - kpt_16)[::-1] * [1, -1]
        else:
            vo = (kpt_0 - kpt_16)[::-1] * [-1, 1]

        # anchor points in the image by eye positions
        pts_tar = np.vstack([kpt_0, kpt_16, kpt_0 + vo, kpt_16 + vo])

        # pts_tar = pose['keypoints'][self.anchor_points_indices]

        h_mat, _ = cv2.findHomography(pts_src, pts_tar)
        patch = cv2.warpPerspective(
            self.src_img,
            h_mat,
            dsize=(canvas.shape[1], canvas.shape[0]),
            borderValue=(255, 255, 255))
        #  mask the white background area in the patch with a threshold 200
        mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask = (mask < 200).astype(np.uint8)
        canvas = cv2.copyTo(patch, mask, canvas)

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
        self.src_img = load_image_from_disk_or_url(src_img_path)

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
class SpriteNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 src_img_path: Optional[str] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        if src_img_path is None:
            # Sprites of Touhou characters :)
            # Come from https://www.deviantart.com/shadowbendy/art/Touhou-rpg-maker-vx-Sprite-1-812746920  # noqa: E501
            src_img_path = (
                'https://user-images.githubusercontent.com/'
                '26739999/151532276-33f968d9-917f-45e3-8a99-ebde60be83bb.png')
        self.src_img = load_image_from_disk_or_url(
            src_img_path, cv2.IMREAD_UNCHANGED)[:144, :108]
        tmp = np.array(np.split(self.src_img, range(36, 144, 36), axis=0))
        tmp = np.array(np.split(tmp, range(36, 108, 36), axis=2))
        self.sprites = tmp
        self.pos = None
        self.anime_frame = 0

    def apply_sprite_effect(self,
                            img,
                            pose_results,
                            left_hand_index,
                            right_hand_index,
                            kpt_thr=0.5):
        """Apply sprite effect.

        Args:
            img (np.ndarray): Image data.
            pose_results (list[dict]): The pose estimation results containing:
                - "keypoints" ([K,3]): detection result in [x, y, score]
            left_hand_index (int): Keypoint index of left hand
            right_hand_index (int): Keypoint index of right hand
            kpt_thr (float): The score threshold of required keypoints.
        """

        hm, wm = self.sprites.shape[2:4]
        # anchor points in the sunglasses mask
        if self.pos is None:
            self.pos = [img.shape[0] // 2, img.shape[1] // 2]

        if len(pose_results) == 0:
            return img

        kpts = pose_results[0]['keypoints']

        if kpts[left_hand_index, 2] < kpt_thr and kpts[right_hand_index,
                                                       2] < kpt_thr:
            aim = self.pos
        else:
            kpt_lhand = kpts[left_hand_index, :2][::-1]
            kpt_rhand = kpts[right_hand_index, :2][::-1]

            def distance(a, b):
                return (a[0] - b[0])**2 + (a[1] - b[1])**2

            # Go to the nearest hand
            if distance(kpt_lhand, self.pos) < distance(kpt_rhand, self.pos):
                aim = kpt_lhand
            else:
                aim = kpt_rhand

        pos_thr = 15
        if aim[0] < self.pos[0] - pos_thr:
            # Go down
            sprite = self.sprites[self.anime_frame][3]
            self.pos[0] -= 1
        elif aim[0] > self.pos[0] + pos_thr:
            # Go up
            sprite = self.sprites[self.anime_frame][0]
            self.pos[0] += 1
        elif aim[1] < self.pos[1] - pos_thr:
            # Go right
            sprite = self.sprites[self.anime_frame][1]
            self.pos[1] -= 1
        elif aim[1] > self.pos[1] + pos_thr:
            # Go left
            sprite = self.sprites[self.anime_frame][2]
            self.pos[1] += 1
        else:
            # Stay
            self.anime_frame = 0
            sprite = self.sprites[self.anime_frame][0]

        if self.anime_frame < 2:
            self.anime_frame += 1
        else:
            self.anime_frame = 0

        x = self.pos[0] - hm // 2
        y = self.pos[1] - wm // 2
        x = max(0, min(x, img.shape[0] - hm))
        y = max(0, min(y, img.shape[0] - wm))

        # Overlay image with transparent
        img[x:x + hm, y:y +
            wm] = (img[x:x + hm, y:y + wm] * (1 - sprite[:, :, 3:] / 255) +
                   sprite[:, :, :3] * (sprite[:, :, 3:] / 255)).astype('uint8')

        return img

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            # left_hand_idx, right_hand_idx = _get_wrist_keypoint_ids(model_cfg)  # noqa: E501
            left_hand_idx, right_hand_idx = _get_eye_keypoint_ids(model_cfg)

            canvas = self.apply_sprite_effect(canvas, preds, left_hand_idx,
                                              right_hand_idx)
        return canvas


@NODES.register_module()
class BackgroundNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 src_img_path: Optional[str] = None,
                 cls_ids: Optional[List] = None,
                 cls_names: Optional[List] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        self.cls_ids = cls_ids
        self.cls_names = cls_names

        if src_img_path is None:
            src_img_path = 'https://user-images.githubusercontent.com/'\
                           '11788150/149731957-abd5c908-9c7f-45b2-b7bf-'\
                           '821ab30c6a3e.jpg'
        self.src_img = load_image_from_disk_or_url(src_img_path)

    def apply_background_effect(self,
                                img,
                                det_results,
                                background_img,
                                effect_region=(0.2, 0.2, 0.8, 0.8)):
        """Change background.

        Args:
            img (np.ndarray): Image data.
            det_results (list[dict]): The detection results containing:

                - "cls_id" (int): Class index.
                - "label" (str): Class label (e.g. 'person').
                - "bbox" (ndarray:(5, )): bounding box result
                    [x, y, w, h, score].
                - "mask" (ndarray:(w, h)): instance segmentation result.
            background_img (np.ndarray): Background image.
            effect_region (tuple(4, )): The region to apply mask,
                the coordinates are normalized (x1, y1, x2, y2).
        """
        if len(det_results) > 0:
            # Choose the one with the highest score.
            det_result = det_results[0]
            bbox = det_result['bbox']
            mask = det_result['mask'].astype(np.uint8)
            img = copy_and_paste(img, background_img, mask, bbox,
                                 effect_region)
            return img
        else:
            return background_img

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        if canvas.shape != self.src_img.shape:
            self.src_img = cv2.resize(self.src_img, canvas.shape[:2])
        det_results = frame_msg.get_detection_results()
        if not det_results:
            return canvas

        full_preds = []
        for det_result in det_results:
            preds = det_result['preds']
            if self.cls_ids:
                # Filter results by class ID
                filtered_preds = [
                    p for p in preds if p['cls_id'] in self.cls_ids
                ]
            elif self.cls_names:
                # Filter results by class name
                filtered_preds = [
                    p for p in preds if p['label'] in self.cls_names
                ]
            else:
                filtered_preds = preds
            full_preds.extend(filtered_preds)

        canvas = self.apply_background_effect(canvas, full_preds, self.src_img)

        return canvas


@NODES.register_module()
class SaiyanNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 hair_img_path: Optional[str] = None,
                 light_video_path: Optional[str] = None,
                 cls_ids: Optional[List] = None,
                 cls_names: Optional[List] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        self.cls_ids = cls_ids
        self.cls_names = cls_names

        if hair_img_path is None:
            hair_img_path = 'https://user-images.githubusercontent.com/'\
                            '11788150/149732117-fcd2d804-dc2c-426c-bee7-'\
                            '94be6146e05c.png'
        self.hair_img = load_image_from_disk_or_url(hair_img_path)

        if light_video_path is None:
            light_video_path = get_cached_file_path(
                'https://'
                'user-images.githubusercontent.com/11788150/149732080'
                '-ea6cfeda-0dc5-4bbb-892a-3831e5580520.mp4')
        self.light_video_path = light_video_path
        self.light_video = cv2.VideoCapture(self.light_video_path)

    def apply_saiyan_effect(self,
                            img,
                            pose_results,
                            saiyan_img,
                            light_frame,
                            face_indices,
                            bbox_thr=0.3,
                            kpt_thr=0.5):
        """Apply saiyan hair effect.

        Args:
            img (np.ndarray): Image data.
            pose_results (list[dict]): The pose estimation results containing:
                - "keypoints" ([K,3]): keypoint detection result
                    in [x, y, score]
            saiyan_img (np.ndarray): Saiyan image with transparent background.
            light_frame (np.ndarray): Light image with green screen.
            face_indices (int): Keypoint index of the face
            kpt_thr (float): The score threshold of required keypoints.
        """
        img = img.copy()
        im_shape = img.shape
        # Apply lightning effects.
        light_mask = screen_matting(light_frame, color='green')

        # anchor points in the mask
        pts_src = np.array(
            [
                [84, 398],  # face kpt 0
                [331, 393],  # face kpt 16
                [84, 145],
                [331, 140]
            ],
            dtype=np.float32)

        for pose in pose_results:
            bbox = pose['bbox']

            if bbox[-1] < bbox_thr:
                continue

            mask_inst = pose['mask']
            # cache
            fg = img[np.where(mask_inst)]

            bbox = expand_and_clamp(bbox[:4], im_shape, s=3.0)
            # Apply light effects between fg and bg
            img = copy_and_paste(
                light_frame,
                img,
                light_mask,
                effect_region=(bbox[0] / im_shape[1], bbox[1] / im_shape[0],
                               bbox[2] / im_shape[1], bbox[3] / im_shape[0]))
            # pop
            img[np.where(mask_inst)] = fg

            # Apply Saiyan hair effects
            kpts = pose['keypoints']
            if kpts[face_indices[0], 2] < kpt_thr or kpts[face_indices[16],
                                                          2] < kpt_thr:
                continue

            kpt_0 = kpts[face_indices[0], :2]
            kpt_16 = kpts[face_indices[16], :2]
            # orthogonal vector
            vo = (kpt_0 - kpt_16)[::-1] * [-1, 1]

            # anchor points in the image by eye positions
            pts_tar = np.vstack([kpt_0, kpt_16, kpt_0 + vo, kpt_16 + vo])

            h_mat, _ = cv2.findHomography(pts_src, pts_tar)
            patch = cv2.warpPerspective(
                saiyan_img,
                h_mat,
                dsize=(img.shape[1], img.shape[0]),
                borderValue=(0, 0, 0))
            mask_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            mask_patch = (mask_patch > 1).astype(np.uint8)
            img = cv2.copyTo(patch, mask_patch, img)

        return img

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()

        det_results = frame_msg.get_detection_results()
        if not det_results:
            return canvas

        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas

        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            face_indices = _get_face_keypoint_ids(model_cfg)

            ret, frame = self.light_video.read()
            if not ret:
                self.light_video = cv2.VideoCapture(self.light_video_path)
                ret, frame = self.light_video.read()

            canvas = self.apply_saiyan_effect(canvas, preds, self.hair_img,
                                              frame, face_indices)

        return canvas


@NODES.register_module()
class MoustacheNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 src_img_path: Optional[str] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        if src_img_path is None:
            src_img_path = 'https://user-images.githubusercontent.com/'\
                           '11788150/149732141-3afbab55-252a-428c-b6d8'\
                           '-0e352f432651.jpeg'
        self.src_img = load_image_from_disk_or_url(src_img_path)

    def apply_moustache_effect(self,
                               img,
                               pose_results,
                               moustache_img,
                               face_indices,
                               kpt_thr=0.5):
        """Apply moustache effect.

        Args:
            img (np.ndarray): Image data.
            pose_results (list[dict]): The pose estimation results containing:
                - "keypoints" ([K,3]): keypoint detection result
                    in [x, y, score]
            moustache_img (np.ndarray): Moustache image with white background.
            left_eye_index (int): Keypoint index of left eye
            right_eye_index (int): Keypoint index of right eye
            kpt_thr (float): The score threshold of required keypoints.
        """

        hm, wm = moustache_img.shape[:2]
        # anchor points in the moustache mask
        pts_src = np.array([[1164, 741], [1729, 741], [1164, 1244],
                            [1729, 1244]],
                           dtype=np.float32)

        for pose in pose_results:
            kpts = pose['keypoints']
            if kpts[face_indices[32], 2] < kpt_thr \
                    or kpts[face_indices[34], 2] < kpt_thr \
                    or kpts[face_indices[61], 2] < kpt_thr \
                    or kpts[face_indices[63], 2] < kpt_thr:
                continue

            kpt_32 = kpts[face_indices[32], :2]
            kpt_34 = kpts[face_indices[34], :2]
            kpt_61 = kpts[face_indices[61], :2]
            kpt_63 = kpts[face_indices[63], :2]
            # anchor points in the image by eye positions
            pts_tar = np.vstack([kpt_32, kpt_34, kpt_61, kpt_63])

            h_mat, _ = cv2.findHomography(pts_src, pts_tar)
            patch = cv2.warpPerspective(
                moustache_img,
                h_mat,
                dsize=(img.shape[1], img.shape[0]),
                borderValue=(255, 255, 255))
            #  mask the white background area in the patch with a threshold 200
            mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            mask = (mask < 200).astype(np.uint8)
            img = cv2.copyTo(patch, mask, img)

        return img

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            face_indices = _get_face_keypoint_ids(model_cfg)
            canvas = self.apply_moustache_effect(canvas, preds, self.src_img,
                                                 face_indices)
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
class NoticeBoardNode(BaseFrameEffectNode):

    default_content_lines = ['This is a notice board!']

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


@NODES.register_module()
class HatNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 src_img_path: Optional[str] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key)

        if src_img_path is None:
            # The image attributes to:
            # http://616pic.com/sucai/1m9i70p52.html
            src_img_path = 'https://user-images.githubusercontent.' \
                           'com/28900607/149766271-2f591c19-9b67-4' \
                           'd92-8f94-c272396ca141.png'
        self.src_img = load_image_from_disk_or_url(src_img_path,
                                                   cv2.IMREAD_UNCHANGED)

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            left_eye_idx, right_eye_idx = _get_eye_keypoint_ids(model_cfg)

            canvas = apply_hat_effect(canvas, preds, self.src_img,
                                      left_eye_idx, right_eye_idx)
        return canvas


@NODES.register_module()
class FirecrackerNode(BaseFrameEffectNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 src_img_path: Optional[str] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key)

        if src_img_path is None:
            self.src_img_path = 'https://user-images.githubusercontent' \
                                '.com/28900607/149766281-6376055c-ed8b' \
                                '-472b-991f-60e6ae6ee1da.gif'
        src_img = cv2.VideoCapture(self.src_img_path)

        self.frame_list = []
        ret, frame = src_img.read()
        while frame is not None:
            self.frame_list.append(frame)
            ret, frame = src_img.read()
        self.num_frames = len(self.frame_list)
        self.frame_idx = 0
        self.frame_period = 4  # each frame in gif lasts for 4 frames in video

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas

        frame = self.frame_list[self.frame_idx // self.frame_period]
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            left_wrist_idx, right_wrist_idx = _get_wrist_keypoint_ids(
                model_cfg)

            canvas = apply_firecracker_effect(canvas, preds, frame,
                                              left_wrist_idx, right_wrist_idx)
        self.frame_idx = (self.frame_idx + 1) % (
            self.num_frames * self.frame_period)

        return canvas
