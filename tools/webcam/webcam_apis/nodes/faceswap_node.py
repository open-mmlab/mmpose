# Copyright (c) OpenMMLab. All rights reserved.
from enum import IntEnum
from typing import List, Union

import cv2
import numpy as np

from mmpose.datasets import DatasetInfo
from .builder import NODES
from .frame_drawing_node import FrameDrawingNode


class Mode(IntEnum):
    NONE = 0,
    SHUFFLE = 1,
    CLONE = 2


@NODES.register_module()
class FaceSwapNode(FrameDrawingNode):

    def __init__(
        self,
        name: str,
        frame_buffer: str,
        output_buffer: Union[str, List[str]],
        mode_key: Union[str, int],
    ):
        super().__init__(name, frame_buffer, output_buffer, enable=True)

        self.mode_key = mode_key
        self.mode_index = 0
        self.register_event(
            self.mode_key, is_keyboard=True, handler_func=self.switch_mode)
        self.history = dict(mode=None)
        self._mode = Mode.SHUFFLE

    @property
    def mode(self):
        return self._mode

    def switch_mode(self):
        """Switch modes by updating mode index."""
        self._mode = Mode((self._mode + 1) % len(Mode))

    def draw(self, frame_msg):

        if self.mode == Mode.NONE:
            self.history = {'mode': Mode.NONE}
            return frame_msg.get_image()

        # Init history
        if self.history['mode'] != self.mode:
            self.history = {'mode': self.mode, 'target_map': {}}

        # Merge pose results
        pose_preds = self._merge_pose_results(frame_msg.get_pose_results())
        num_target = len(pose_preds)

        # Show mode
        img = frame_msg.get_image()
        canvas = img.copy()
        if self.mode == Mode.SHUFFLE:
            mode_txt = 'Shuffle'
        else:
            mode_txt = 'Clone'

        cv2.putText(canvas, mode_txt, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (255, 126, 0), 1)

        # Skip if target number is less than 2
        if num_target >= 2:
            # Generate new mapping if target number changes
            if num_target != len(self.history['target_map']):
                if self.mode == Mode.SHUFFLE:
                    self.history['target_map'] = self._get_swap_map(num_target)
                else:
                    self.history['target_map'] = np.repeat(
                        np.random.choice(num_target), num_target)

            # # Draw on canvas
            for tar_idx, src_idx in enumerate(self.history['target_map']):
                face_src = self._get_face_info(pose_preds[src_idx])
                face_tar = self._get_face_info(pose_preds[tar_idx])
                canvas = self._swap_face(img, canvas, face_src, face_tar)

        return canvas

    def _crop_face_by_contour(self, img, contour):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 1)
        mask = cv2.dilate(
            mask, kernel=np.ones((9, 9), dtype=np.uint8), anchor=(4, 0))
        x1, y1, w, h = cv2.boundingRect(mask)
        x2 = x1 + w
        y2 = y1 + h
        bbox = np.array([x1, y1, x2, y2], dtype=np.int64)
        patch = img[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

        return bbox, patch, mask

    def _swap_face(self, img_src, img_tar, face_src, face_tar):

        if face_src['dataset'] == face_tar['dataset']:
            # Use full keypoints for face alignment
            kpts_src = face_src['contour']
            kpts_tar = face_tar['contour']
        else:
            # Use only common landmarks (eyes and nose) for face alignment if
            # source and target have differenet data type
            # (e.g. human vs animal)
            kpts_src = face_src['landmarks']
            kpts_tar = face_tar['landmarks']

        # Get everything local
        bbox_src, patch_src, mask_src = self._crop_face_by_contour(
            img_src, face_src['contour'])

        bbox_tar, _, mask_tar = self._crop_face_by_contour(
            img_tar, face_tar['contour'])

        kpts_src = kpts_src - bbox_src[:2]
        kpts_tar = kpts_tar - bbox_tar[:2]

        # Compute affine transformation matrix
        trans_mat, _ = cv2.estimateAffine2D(
            kpts_src.astype(np.float32), kpts_tar.astype(np.float32))
        patch_warp = cv2.warpAffine(
            patch_src,
            trans_mat,
            dsize=tuple(bbox_tar[2:] - bbox_tar[:2]),
            borderValue=(0, 0, 0))
        mask_warp = cv2.warpAffine(
            mask_src,
            trans_mat,
            dsize=tuple(bbox_tar[2:] - bbox_tar[:2]),
            borderValue=(0, 0, 0))

        # Target mask
        mask_tar = mask_tar & mask_warp
        mask_tar_soft = cv2.GaussianBlur(mask_tar * 255, (3, 3), 3)

        # Blending
        center = tuple((0.5 * (bbox_tar[:2] + bbox_tar[2:])).astype(np.int64))
        img_tar = cv2.seamlessClone(patch_warp, img_tar, mask_tar_soft, center,
                                    cv2.NORMAL_CLONE)
        return img_tar

    @staticmethod
    def _get_face_info(pose_pred):
        keypoints = pose_pred['keypoints'][:, :2]
        model_cfg = pose_pred['model_cfg']
        dataset_info = DatasetInfo(model_cfg.data.test.dataset_info)

        face_info = {
            'dataset': dataset_info.dataset_name,
            'landmarks': None,  # For alignment
            'contour': None,  # For mask generation
            'bbox': None  # For image warping
        }

        # Fall back to hard coded keypoint id

        if face_info['dataset'] == 'coco':
            face_info['landmarks'] = np.stack([
                keypoints[1],  # left eye
                keypoints[2],  # right eye
                keypoints[0],  # nose
                0.5 * (keypoints[5] + keypoints[6]),  # neck (shoulder center)
            ])
        elif face_info['dataset'] == 'coco_wholebody':
            face_info['landmarks'] = np.stack([
                keypoints[1],  # left eye
                keypoints[2],  # right eye
                keypoints[0],  # nose
                keypoints[32],  # chin
            ])
            contour_ids = list(range(23, 40)) + list(range(40, 50))[::-1]
            face_info['contour'] = keypoints[contour_ids]
        elif face_info['dataset'] == 'ap10k':
            face_info['landmarks'] = np.stack([
                keypoints[0],  # left eye
                keypoints[1],  # right eye
                keypoints[2],  # nose
                keypoints[3],  # neck
            ])
        elif face_info['dataset'] == 'animalpose':
            face_info['landmarks'] = np.stack([
                keypoints[0],  # left eye
                keypoints[1],  # right eye
                keypoints[4],  # nose
                keypoints[5],  # throat
            ])
        elif face_info['dataset'] == 'wflw':
            face_info['landmarks'] = np.stack([
                keypoints[97],  # left eye
                keypoints[96],  # right eye
                keypoints[54],  # nose
                keypoints[16],  # chine
            ])
            contour_ids = list(range(33))[::-1] + list(range(33, 38)) + list(
                range(42, 47))
            face_info['contour'] = keypoints[contour_ids]
        else:
            raise ValueError('Can not obtain face landmark information'
                             f'from dataset: {face_info["type"]}')

        # Face region
        if face_info['contour'] is None:
            # Manually defined counter of face region
            left_eye, right_eye, nose = face_info['landmarks'][:3]
            eye_center = 0.5 * (left_eye + right_eye)
            w_vec = right_eye - left_eye
            eye_dist = np.linalg.norm(w_vec) + 1e-6
            w_vec = w_vec / eye_dist
            h_vec = np.array([w_vec[1], -w_vec[0]], dtype=w_vec.dtype)
            w = max(0.5 * eye_dist, np.abs(np.dot(nose - eye_center, w_vec)))
            h = np.abs(np.dot(nose - eye_center, h_vec))

            left_top = eye_center + 1.5 * w * w_vec - 0.5 * h * h_vec
            right_top = eye_center - 1.5 * w * w_vec - 0.5 * h * h_vec
            left_bottom = eye_center + 1.5 * w * w_vec + 4 * h * h_vec
            right_bottom = eye_center - 1.5 * w * w_vec + 4 * h * h_vec

            face_info['contour'] = np.stack(
                [left_top, right_top, right_bottom, left_bottom])

        # Get tight bbox of face region
        face_info['bbox'] = np.array([
            face_info['contour'][:, 0].min(), face_info['contour'][:, 1].min(),
            face_info['contour'][:, 0].max(), face_info['contour'][:, 1].max()
        ]).astype(np.int64)

        return face_info

    @staticmethod
    def _merge_pose_results(pose_results):
        preds = []
        if pose_results is not None:
            for prefix, pose_result in enumerate(pose_results):
                model_cfg = pose_result['model_cfg']
                for idx, _pred in enumerate(pose_result['preds']):
                    pred = _pred.copy()
                    pred['id'] = f'{prefix}.{_pred.get("track_id", str(idx))}'
                    pred['model_cfg'] = model_cfg
                    preds.append(pred)
        return preds

    @staticmethod
    def _get_swap_map(num_target):
        ids = np.random.choice(num_target, num_target, replace=False)
        target_map = ids[(ids + 1) % num_target]
        return target_map
