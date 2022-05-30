# Copyright (c) OpenMMLab. All rights reserved.
import copy
from dataclasses import dataclass
from typing import List, Optional, Union

from .builder import NODES
from .node import Node

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import numpy as np

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model)
from mmpose.core import Smoother


@dataclass
class TrackInfo:
    next_id: int = 0
    last_pose_preds: List = None


@NODES.register_module()
class PoseTrackerNode(Node):

    def __init__(
            self,
            name: str,
            det_model_config: str,
            det_model_checkpoint: str,
            pose_model_config: str,
            pose_model_checkpoint: str,
            input_buffer: str,
            output_buffer: Union[str, List[str]],
            enable_key: Optional[Union[str, int]] = None,
            enable: bool = True,
            device: str = 'cuda:0',
            det_interval: int = 1,
            cls_ids: Optional[List] = None,
            cls_names: Optional[List] = None,
            bbox_thr: float = 0.5,
            kpt2bbox_cfg: dict = {},
            smooth: bool = False,
            smooth_filter_cfg: str = 'configs/_base_/filters/one_euro.py'):

        assert has_mmdet, \
            f'MMDetection is required for {self.__class__.__name__}.'

        super().__init__(name=name, enable_key=enable_key, enable=enable)

        self.det_model_config = det_model_config
        self.det_model_checkpoint = det_model_checkpoint
        self.pose_model_config = pose_model_config
        self.pose_model_checkpoint = pose_model_checkpoint
        self.device = device.lower()
        self.cls_ids = cls_ids
        self.cls_names = cls_names
        self.bbox_thr = bbox_thr
        self.smooth = smooth
        self.smooth_filter_cfg = smooth_filter_cfg
        self.det_interval = det_interval
        self.kpt2bbox_cfg = copy.deepcopy(kpt2bbox_cfg)

        self.det_countdown = 0
        self.track_info = TrackInfo()
        if smooth:
            self.smoother = Smoother(smooth_filter_cfg, keypoint_dim=2)
        else:
            self.smoother = None

        # init models
        self.det_model = init_detector(
            self.det_model_config,
            self.det_model_checkpoint,
            device=self.device)

        self.pose_model = init_pose_model(
            self.pose_model_config,
            self.pose_model_checkpoint,
            device=self.device)

        # register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def bypass(self, input_msgs):
        return input_msgs['input']

    def process(self, input_msgs):
        input_msg = input_msgs['input']
        img = input_msg.get_image()

        if self.det_countdown == 0:
            # get objects by detection model
            self.det_countdown = self.det_interval
            preds = inference_detector(self.det_model, img)
            det_results = self._post_process_det(preds)
        else:
            # get object by pose tracking
            det_results = self._get_objects_from_tracking(img.shape)

        self.det_countdown -= 1

        pose_preds, _ = inference_top_down_pose_model(
            self.pose_model,
            img,
            det_results['preds'],
            bbox_thr=self.bbox_thr,
            format='xyxy')

        pose_preds, next_id = get_track_id(
            pose_preds,
            self.track_info.last_pose_preds,
            self.track_info.next_id,
            use_oks=False,
            tracking_thr=0.3)

        self.track_info.next_id = next_id
        self.track_info.last_pose_preds = pose_preds.copy()

        # Pose smoothing
        if self.smoother:
            pose_preds = self.smoother.smooth(pose_preds)

        pose_result = {
            'preds': pose_preds,
            'model_cfg': self.pose_model.cfg.copy(),
        }

        input_msg.add_detection_result(det_results, tag=self.name)
        input_msg.add_pose_result(pose_result, tag=self.name)

        return input_msg

    def _get_objects_from_tracking(self, img_shape):

        det_results = {'preds': [], 'model_cfg': self.det_model.cfg.copy()}
        for pred in self.track_info.last_pose_preds:
            pred = copy.deepcopy(pred)
            kpts = pred.pop('keypoints')
            bbox = self._keypoints_to_bbox(kpts, img_shape)
            if bbox is not None:
                pred['bbox'][:4] = bbox
            det_results['preds'].append(pred)

        return det_results

    def _keypoints_to_bbox(self, keypoints, img_shape):

        scale = self.kpt2bbox_cfg.get('scale', 1.5)
        kpt_thr = self.kpt2bbox_cfg.get('kpt_thr', 0.3)
        valid = keypoints[:, 2] > kpt_thr

        if not valid.any():
            return None

        x1 = np.min(keypoints[valid, 0])
        y1 = np.min(keypoints[valid, 1])
        x2 = np.max(keypoints[valid, 0])
        y2 = np.max(keypoints[valid, 1])

        xc = 0.5 * (x1 + x2)
        yc = 0.5 * (y1 + y2)
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale

        img_h, img_w = img_shape[:2]

        bbox = np.array([
            np.clip(0, img_w, xc - 0.5 * w),
            np.clip(0, img_h, yc - 0.5 * h),
            np.clip(0, img_w, xc + 0.5 * w),
            np.clip(0, img_h, yc + 0.5 * h)
        ]).astype(np.float32)
        return bbox

    def _post_process_det(self, preds):
        if isinstance(preds, tuple):
            dets = preds[0]
            segms = preds[1]
        else:
            dets = preds
            segms = [None] * len(dets)

        det_model_classes = self.det_model.CLASSES
        if isinstance(det_model_classes, str):
            det_model_classes = (det_model_classes, )

        assert len(dets) == len(det_model_classes)
        assert len(segms) == len(det_model_classes)
        result = {'preds': [], 'model_cfg': self.det_model.cfg.copy()}

        for i, (cls_name, bboxes,
                masks) in enumerate(zip(det_model_classes, dets, segms)):
            if masks is None:
                masks = [None] * len(bboxes)
            else:
                assert len(masks) == len(bboxes)

            preds_i = [{
                'cls_id': i,
                'label': cls_name,
                'bbox': bbox,
                'mask': mask,
            } for (bbox, mask) in zip(bboxes, masks)]

            if self.cls_ids:
                preds_i = [
                    pred for pred in preds_i if pred['cls_id'] in self.cls_ids
                ]
            elif self.cls_names:
                preds_i = [
                    pred for pred in preds_i if pred['label'] in self.cls_names
                ]

            result['preds'].extend(preds_i)

        return result
