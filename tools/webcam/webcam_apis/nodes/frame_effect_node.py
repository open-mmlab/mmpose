# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv import color_val

from mmpose.core import (apply_bugeye_effect, apply_sunglasses_effect,
                         imshow_bboxes, imshow_keypoints)
from mmpose.datasets import DatasetInfo
from ..utils import (FrameMessage, copy_and_paste, expand_and_clamp,
                     get_cached_file_path, get_eye_keypoint_ids,
                     get_face_keypoint_ids, get_wrist_keypoint_ids,
                     load_image_from_disk_or_url, screen_matting)
from .builder import NODES
from .frame_drawing_node import FrameDrawingNode

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


@NODES.register_module()
class PoseVisualizerNode(FrameDrawingNode):
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
class SunglassesNode(FrameDrawingNode):

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
            left_eye_idx, right_eye_idx = get_eye_keypoint_ids(model_cfg)

            canvas = apply_sunglasses_effect(canvas, preds, self.src_img,
                                             left_eye_idx, right_eye_idx)
        return canvas


@NODES.register_module()
class SpriteNode(FrameDrawingNode):

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
            # left_hand_idx, right_hand_idx = get_wrist_keypoint_ids(model_cfg)  # noqa: E501
            left_hand_idx, right_hand_idx = get_eye_keypoint_ids(model_cfg)

            canvas = self.apply_sprite_effect(canvas, preds, left_hand_idx,
                                              right_hand_idx)
        return canvas


@NODES.register_module()
class BackgroundNode(FrameDrawingNode):

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
class SaiyanNode(FrameDrawingNode):

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
            face_indices = get_face_keypoint_ids(model_cfg)

            ret, frame = self.light_video.read()
            if not ret:
                self.light_video = cv2.VideoCapture(self.light_video_path)
                ret, frame = self.light_video.read()

            canvas = self.apply_saiyan_effect(canvas, preds, self.hair_img,
                                              frame, face_indices)

        return canvas


@NODES.register_module()
class MoustacheNode(FrameDrawingNode):

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
            face_indices = get_face_keypoint_ids(model_cfg)
            canvas = self.apply_moustache_effect(canvas, preds, self.src_img,
                                                 face_indices)
        return canvas


@NODES.register_module()
class BugEyeNode(FrameDrawingNode):

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            left_eye_idx, right_eye_idx = get_eye_keypoint_ids(model_cfg)

            canvas = apply_bugeye_effect(canvas, preds, left_eye_idx,
                                         right_eye_idx)
        return canvas


@NODES.register_module()
class NoticeBoardNode(FrameDrawingNode):

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
class HatNode(FrameDrawingNode):

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

    @staticmethod
    def apply_hat_effect(img,
                         pose_results,
                         hat_img,
                         left_eye_index,
                         right_eye_index,
                         kpt_thr=0.5):
        """Apply hat effect.
        Args:
            img (np.ndarray): Image data.
            pose_results (list[dict]): The pose estimation results containing:
                - "keypoints" ([K,3]): keypoint detection result in
                    [x, y, score]
            hat_img (np.ndarray): Hat image with white alpha channel.
            left_eye_index (int): Keypoint index of left eye
            right_eye_index (int): Keypoint index of right eye
            kpt_thr (float): The score threshold of required keypoints.
        """
        img_orig = img.copy()

        img = img_orig.copy()
        hm, wm = hat_img.shape[:2]
        # anchor points in the sunglasses mask
        a = 0.3
        b = 0.7
        pts_src = np.array([[a * wm, a * hm], [a * wm, b * hm],
                            [b * wm, a * hm], [b * wm, b * hm]],
                           dtype=np.float32)

        for pose in pose_results:
            kpts = pose['keypoints']

            if kpts[left_eye_index, 2] < kpt_thr or \
                    kpts[right_eye_index, 2] < kpt_thr:
                continue

            kpt_leye = kpts[left_eye_index, :2]
            kpt_reye = kpts[right_eye_index, :2]
            # orthogonal vector to the left-to-right eyes
            vo = 0.5 * (kpt_reye - kpt_leye)[::-1] * [-1, 1]
            veye = 0.5 * (kpt_reye - kpt_leye)

            # anchor points in the image by eye positions
            pts_tar = np.vstack([
                kpt_reye + 1 * veye + 5 * vo, kpt_reye + 1 * veye + 1 * vo,
                kpt_leye - 1 * veye + 5 * vo, kpt_leye - 1 * veye + 1 * vo
            ])

            h_mat, _ = cv2.findHomography(pts_src, pts_tar)
            patch = cv2.warpPerspective(
                hat_img,
                h_mat,
                dsize=(img.shape[1], img.shape[0]),
                borderValue=(255, 255, 255))
            #  mask the white background area in the patch with a threshold 200
            mask = (patch[:, :, -1] > 128)
            patch = patch[:, :, :-1]
            mask = mask * (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) > 30)
            mask = mask.astype(np.uint8)

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
            left_eye_idx, right_eye_idx = get_eye_keypoint_ids(model_cfg)

            canvas = self.apply_hat_effect(canvas, preds, self.src_img,
                                           left_eye_idx, right_eye_idx)
        return canvas


@NODES.register_module()
class FirecrackerNode(FrameDrawingNode):

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

    @staticmethod
    def apply_firecracker_effect(img,
                                 pose_results,
                                 firecracker_img,
                                 left_wrist_idx,
                                 right_wrist_idx,
                                 kpt_thr=0.5):
        """Apply firecracker effect.
        Args:
            img (np.ndarray): Image data.
            pose_results (list[dict]): The pose estimation results containing:
                - "keypoints" ([K,3]): keypoint detection result in
                    [x, y, score]
            firecracker_img (np.ndarray): Firecracker image with white
                background.
            left_wrist_idx (int): Keypoint index of left wrist
            right_wrist_idx (int): Keypoint index of right wrist
            kpt_thr (float): The score threshold of required keypoints.
        """

        hm, wm = firecracker_img.shape[:2]
        # anchor points in the firecracker mask
        pts_src = np.array([[0. * wm, 0. * hm], [0. * wm, 1. * hm],
                            [1. * wm, 0. * hm], [1. * wm, 1. * hm]],
                           dtype=np.float32)

        h, w = img.shape[:2]
        h_tar = h / 3
        w_tar = h_tar / hm * wm

        for pose in pose_results:
            kpts = pose['keypoints']

            if kpts[left_wrist_idx, 2] > kpt_thr:
                kpt_lwrist = kpts[left_wrist_idx, :2]
                # anchor points in the image by eye positions
                pts_tar = np.vstack([
                    kpt_lwrist - [w_tar / 2, 0],
                    kpt_lwrist - [w_tar / 2, -h_tar],
                    kpt_lwrist + [w_tar / 2, 0],
                    kpt_lwrist + [w_tar / 2, h_tar]
                ])

                h_mat, _ = cv2.findHomography(pts_src, pts_tar)
                patch = cv2.warpPerspective(
                    firecracker_img,
                    h_mat,
                    dsize=(img.shape[1], img.shape[0]),
                    borderValue=(255, 255, 255))
                #  mask the white background area in the patch with
                # a threshold 200
                mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                mask = (mask < 240).astype(np.uint8)
                img = cv2.copyTo(patch, mask, img)

            if kpts[right_wrist_idx, 2] > kpt_thr:
                kpt_rwrist = kpts[right_wrist_idx, :2]

                # anchor points in the image by eye positions
                pts_tar = np.vstack([
                    kpt_rwrist - [w_tar / 2, 0],
                    kpt_rwrist - [w_tar / 2, -h_tar],
                    kpt_rwrist + [w_tar / 2, 0],
                    kpt_rwrist + [w_tar / 2, h_tar]
                ])

                h_mat, _ = cv2.findHomography(pts_src, pts_tar)
                patch = cv2.warpPerspective(
                    firecracker_img,
                    h_mat,
                    dsize=(img.shape[1], img.shape[0]),
                    borderValue=(255, 255, 255))
                #  mask the white background area in the patch with
                # a threshold 200
                mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                mask = (mask < 240).astype(np.uint8)
                img = cv2.copyTo(patch, mask, img)

        return img

    def draw(self, frame_msg):
        canvas = frame_msg.get_image()
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas

        frame = self.frame_list[self.frame_idx // self.frame_period]
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            left_wrist_idx, right_wrist_idx = get_wrist_keypoint_ids(model_cfg)

            canvas = self.apply_firecracker_effect(canvas, preds, frame,
                                                   left_wrist_idx,
                                                   right_wrist_idx)
        self.frame_idx = (self.frame_idx + 1) % (
            self.num_frames * self.frame_period)

        return canvas
