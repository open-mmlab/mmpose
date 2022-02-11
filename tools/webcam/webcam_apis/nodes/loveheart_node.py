# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..utils import (FrameMessage, get_flying_heart_keypoint_ids,
                     get_hand_heart_keypoint_ids, get_mouth_keypoint_ids,
                     load_image_from_disk_or_url)
from .builder import NODES
from .frame_drawing_node import FrameDrawingNode


@dataclass
class LoveHeartInfo():
    """Dataclass for love heart information."""
    heart_type: int
    start_time: float
    start_pos: List[int]
    end_pos: List[int]


@NODES.register_module()
class LoveHeartNode(FrameDrawingNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 kpt_vis_thr: float = 0.3,
                 hand_heart_angle_thr: float = 90.0,
                 longest_duration: float = 2.0,
                 largest_size: Tuple[int, int] = (144, 144),
                 hand_heart_img_path: Optional[str] = None,
                 flying_heart_img_path: Optional[str] = None,
                 hand_heart_dis_thr: float = 50.0,
                 flying_heart_dis_thr: float = 50.0):

        super().__init__(
            name, frame_buffer, output_buffer, enable_key=enable_key)

        if hand_heart_img_path is None:
            hand_heart_img_path = 'https://user-images.githubusercontent.com/'\
                           '87690686/149731850-ea946766-a4e8-4efa-82f5'\
                           '-e2f0515db8ae.png'
        if flying_heart_img_path is None:
            flying_heart_img_path = 'https://user-images.githubusercontent.'\
                                    'com/87690686/153554948-937ce496-33dd-4'\
                                    '9ab-9829-0433fd7c13c4.png'

        self.hand_heart = load_image_from_disk_or_url(hand_heart_img_path)
        self.flying_heart = load_image_from_disk_or_url(flying_heart_img_path)

        self.kpt_vis_thr = kpt_vis_thr
        self.hand_heart_angle_thr = hand_heart_angle_thr
        self.hand_heart_dis_thr = hand_heart_dis_thr
        self.flying_heart_dis_thr = flying_heart_dis_thr
        self.longest_duration = longest_duration
        self.largest_size = largest_size

        # record the love heart infos for each person
        self.heart_infos = {}

    def _cal_distance(self, p1: np.ndarray, p2: np.ndarray) -> np.float64:
        """calculate the distance of points p1 and p2."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _cal_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                   p4: np.ndarray) -> np.float64:
        """calculate the angle of vectors v1(constructed by points p2 and p1)
        and v2(constructed by points p4 and p3)"""
        v1 = p2 - p1
        v2 = p4 - p3

        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = np.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * np.sqrt(
            pow(v2[0], 2) + pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)

        return (np.arccos(cos) / np.pi) * 180

    def _check_heart(self, pred: Dict[str, np.ndarray],
                     hand_heart_indices: List[int],
                     flying_heart_indices: List[int],
                     mouth_indices: List[int]) -> int:
        """Check the type of love heart based on the pose results and keypoint
        indices of hand and mouth.

        Args:
            pred(dict): The pose estimation results containing:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result
                                                 in [x, y, score]
            hand_heart_indices(list[int]): keypoint indices of hand heart
            flying_heart_indices(list[int]): keypoint indices of flying heart

        Returns:
            int: a number representing the type of love heart pose,
                 0: None, 1: hand heart, 2: left hand blow kiss,
                 3: right hand blow kiss
        """
        kpts = pred['keypoints']

        for i in hand_heart_indices:
            if kpts[i][2] < self.kpt_vis_thr:
                return 0

        p1 = kpts[hand_heart_indices[3]][:2]
        p2 = kpts[hand_heart_indices[2]][:2]
        p3 = kpts[hand_heart_indices[1]][:2]
        p4 = kpts[hand_heart_indices[0]][:2]
        left_angle = self._cal_angle(p1, p2, p3, p4)

        p1 = kpts[hand_heart_indices[7]][:2]
        p2 = kpts[hand_heart_indices[6]][:2]
        p3 = kpts[hand_heart_indices[5]][:2]
        p4 = kpts[hand_heart_indices[4]][:2]
        right_angle = self._cal_angle(p1, p2, p3, p4)

        dis = self._cal_distance(kpts[hand_heart_indices[3]][:2],
                                 kpts[hand_heart_indices[7]][:2])

        if left_angle < self.hand_heart_angle_thr and \
           right_angle < self.hand_heart_angle_thr and \
           dis < self.hand_heart_dis_thr:
            return 1

        left_hand_vis = True
        for i in flying_heart_indices[:2]:
            if kpts[i][2] < self.kpt_vis_thr:
                left_hand_vis = False
                break

        right_hand_vis = True
        for i in flying_heart_indices[2:]:
            if kpts[i][2] < self.kpt_vis_thr:
                right_hand_vis = False
                break

        mouth_vis = True
        for i in mouth_indices:
            if kpts[i][2] < self.kpt_vis_thr:
                mouth_vis = False
                break

        if (not left_hand_vis and not right_hand_vis) or not mouth_vis:
            return 0

        left_mouth_idx, right_mouth_idx = mouth_indices
        mouth_pos = (kpts[left_mouth_idx][:2] + kpts[right_mouth_idx][:2]) / 2

        left_mid_hand_pos = (kpts[flying_heart_indices[0]][:2] +
                             kpts[flying_heart_indices[1]][:2]) / 2
        dis = self._cal_distance(left_mid_hand_pos, mouth_pos)
        if dis < self.flying_heart_dis_thr:
            return 2

        right_mid_hand_pos = (kpts[flying_heart_indices[2]][:2] +
                              kpts[flying_heart_indices[3]][:2]) / 2
        dis = self._cal_distance(right_mid_hand_pos, mouth_pos)

        if dis < self.flying_heart_dis_thr:
            return 3

        return 0

    def _get_heart_route(self, heart_type: int, cur_pred: Dict[str,
                                                               np.ndarray],
                         tar_pred: Dict[str, np.ndarray],
                         hand_heart_indices: List[int],
                         flying_heart_indices: List[int],
                         mouth_indices: List[int]) -> Tuple[int, int]:
        """get the start and end position of the love heart, based on two
        keypoint results and keypoint indices of hand and mouth.

        Args:
            cur_pred(dict): The pose estimation results of current person,
                containing: the following keys:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result
                                                 in [x, y, score]
            tar_pred(dict): The pose estimation results of target person,
                containing: the following keys:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result
                                                 in [x, y, score]
            hand_heart_indices(list[int]): keypoint indices of hand heart
            flying_heart_indices(list[int]): keypoint indices of flying heart
            mouth_indices(list[int]): keypoint indices of mouth

        Returns:
            tuple(int): the start position of love heart
            tuple(int): the end position of love heart
        """
        cur_kpts = cur_pred['keypoints']

        assert heart_type in [1, 2,
                              3], 'Can not determine the type of heart effect'

        if heart_type == 1:
            p1 = cur_kpts[hand_heart_indices[3]][:2]
            p2 = cur_kpts[hand_heart_indices[7]][:2]
        elif heart_type == 2:
            p1 = cur_kpts[flying_heart_indices[0]][:2]
            p2 = cur_kpts[flying_heart_indices[1]][:2]
        elif heart_type == 3:
            p1 = cur_kpts[flying_heart_indices[2]][:2]
            p2 = cur_kpts[flying_heart_indices[3]][:2]

        cur_x, cur_y = (p1 + p2) / 2
        # the mid point of two fingers
        start_pos = [int(cur_x), int(cur_y)]

        tar_kpts = tar_pred['keypoints']
        left_mouth_idx, right_mouth_idx = mouth_indices

        p3 = tar_kpts[left_mouth_idx][:2]
        p4 = tar_kpts[right_mouth_idx][:2]
        tar_x, tar_y = (p3 + p4) / 2
        # the mid point of two keypoints
        end_pos = [int(tar_x), int(tar_y)]

        return start_pos, end_pos

    def _draw_heart(self, canvas: np.ndarray, heart_info: LoveHeartInfo,
                    t_pass: float) -> np.ndarray:
        """draw the love heart according to heart info and time."""
        start_x, start_y = heart_info.start_pos
        end_x, end_y = heart_info.end_pos

        scale = t_pass / self.longest_duration
        hm, wm = self.largest_size
        new_h, new_w = int(hm * scale), int(wm * scale)
        max_h, max_w = canvas.shape[:2]

        x = int(start_x + scale * (end_x - start_x))
        y = int(start_y + scale * (end_y - start_y))

        y1 = max(0, y - int(new_h / 2))
        y2 = min(max_h - 1, y + int(new_h / 2))

        x1 = max(0, x - int(new_w / 2))
        x2 = min(max_w - 1, x + int(new_w / 2))

        target = canvas[y1:y2 + 1, x1:x2 + 1].copy()
        new_h, new_w = target.shape[:2]

        if new_h == 0 or new_w == 0:
            return canvas

        assert heart_info.heart_type in [
            1, 2, 3
        ], 'Can not determine the type of heart effect'
        if heart_info.heart_type == 1:  # hand heart
            patch = self.hand_heart.copy()
        elif heart_info.heart_type >= 2:  # hand blow kiss
            patch = self.flying_heart.copy()

        patch = cv2.resize(patch, (new_w, new_h))
        mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask = (mask < 100)[..., None].astype(np.float32) * 0.8

        canvas[y1:y2 + 1, x1:x2 + 1] = patch * mask + target * (1 - mask)

        return canvas

    def draw(self, frame_msg: FrameMessage) -> np.ndarray:
        canvas = frame_msg.get_image()

        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas

        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']

            preds = [pred.copy() for pred in pose_result['preds']]
            # if number of persons in the image is less than 2,
            # no love heart effect will be triggered
            if len(preds) < 2:
                continue

            # if number of persons in the image is more than 2,
            # only use the first two pose results
            preds = preds[:2]
            ids = [preds[i]['track_id'] for i in range(2)]

            for id in self.heart_infos.copy():
                if id not in ids:
                    # if the id of a person not in previous heart_infos,
                    # delete the corresponding field
                    del self.heart_infos[id]

            for i in range(2):
                id = preds[i]['track_id']

                # if the predicted person in previous heart_infos,
                # draw the love heart
                if id in self.heart_infos.copy():
                    t_pass = time.time() - self.heart_infos[id].start_time

                    # the time passed since last love heart pose less than
                    # longest_duration, continue to draw the heart
                    if t_pass < self.longest_duration:
                        canvas = self._draw_heart(canvas, self.heart_infos[id],
                                                  t_pass)
                    # reset corresponding heart info
                    else:
                        del self.heart_infos[id]
                else:
                    # the related keypoint indices of hand heart, e.g,
                    # get the indices of the following keypoints:
                    # left_hand_root, left_pinky_finger1,
                    # left_pinky_finger3, left_pinky_finger4,
                    # right_hand_root, right_pinky_finger1
                    # right_pinky_finger3, right_pinky_finger4
                    hand_heart_indices = get_hand_heart_keypoint_ids(model_cfg)

                    # the related keypoint indices of flying heart, e.g,
                    # get the indices of the following keypoints:
                    # left_middle_finger1, left_middle_finger4,
                    # right_middle_finger1, right_middle_finger4
                    flying_heart_indices = get_flying_heart_keypoint_ids(
                        model_cfg)

                    # hand_indices = get_hand_keypoint_ids(model_cfg)
                    mouth_indices = get_mouth_keypoint_ids(model_cfg)

                    # Check the type of love heart based on the pose results
                    # and keypoint indices of hand and mouth
                    heart_type = self._check_heart(preds[i],
                                                   hand_heart_indices,
                                                   flying_heart_indices,
                                                   mouth_indices)
                    # trigger a love heart effect
                    if heart_type:
                        # get the route of love heart
                        start_pos, end_pos = self._get_heart_route(
                            heart_type, preds[i], preds[1 - i],
                            hand_heart_indices, flying_heart_indices,
                            mouth_indices)
                        start_time = time.time()
                        self.heart_infos[id] = LoveHeartInfo(
                            heart_type, start_time, start_pos, end_pos)

        return canvas
