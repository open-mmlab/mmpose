# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from mmpose.apis import (get_track_id, inference_gesture_model,
                         inference_top_down_pose_model, init_pose_model)
from mmpose.core import Smoother
from ..utils import Message
from .builder import NODES
from .node import MultiInputNode, Node


@dataclass
class TrackInfo:
    next_id: int = 0
    last_pose_preds: List = None


@NODES.register_module()
class TopDownPoseEstimatorNode(Node):

    def __init__(
            self,
            name: str,
            model_config: str,
            model_checkpoint: str,
            input_buffer: str,
            output_buffer: Union[str, List[str]],
            enable_key: Optional[Union[str, int]] = None,
            enable: bool = True,
            device: str = 'cuda:0',
            cls_ids: Optional[List] = None,
            cls_names: Optional[List] = None,
            bbox_thr: float = 0.5,
            smooth: bool = False,
            smooth_filter_cfg: str = 'configs/_base_/filters/one_euro.py'):
        super().__init__(name=name, enable_key=enable_key, enable=enable)

        # Init model
        self.model_config = model_config
        self.model_checkpoint = model_checkpoint
        self.device = device.lower()

        self.cls_ids = cls_ids
        self.cls_names = cls_names
        self.bbox_thr = bbox_thr

        if smooth:
            self.smoother = Smoother(smooth_filter_cfg, keypoint_dim=2)
        else:
            self.smoother = None
        # Init model
        self.model = init_pose_model(
            self.model_config, self.model_checkpoint, device=self.device)

        # Store history for pose tracking
        self.track_info = TrackInfo()

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', trigger=True)
        self.register_output_buffer(output_buffer)

    def bypass(self, input_msgs):
        return input_msgs['input']

    def process(self, input_msgs):

        input_msg = input_msgs['input']
        img = input_msg.get_image()
        det_results = input_msg.get_detection_results()

        if det_results is None:
            raise ValueError(
                'No detection results are found in the frame message.'
                f'{self.__class__.__name__} should be used after a '
                'detector node.')

        full_det_preds = []
        for det_result in det_results:
            det_preds = det_result['preds']
            if self.cls_ids:
                # Filter detection results by class ID
                det_preds = [
                    p for p in det_preds if p['cls_id'] in self.cls_ids
                ]
            elif self.cls_names:
                # Filter detection results by class name
                det_preds = [
                    p for p in det_preds if p['label'] in self.cls_names
                ]
            full_det_preds.extend(det_preds)

        # Inference pose
        pose_preds, _ = inference_top_down_pose_model(
            self.model,
            img,
            full_det_preds,
            bbox_thr=self.bbox_thr,
            format='xyxy')

        # Pose tracking
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
            'model_cfg': self.model.cfg.copy(),
        }

        input_msg.add_pose_result(pose_result, tag=self.name)

        return input_msg


@NODES.register_module()
class HandGestureRecognizerNode(TopDownPoseEstimatorNode, MultiInputNode):

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 device: str = 'cuda:0',
                 cls_ids: Optional[List] = None,
                 cls_names: Optional[List] = None,
                 bbox_thr: float = 0.5,
                 min_frame: int = 16,
                 fps: int = 30,
                 score_thr: float = 0.7):
        TopDownPoseEstimatorNode.__init__(
            self,
            name=name,
            model_config=model_config,
            model_checkpoint=model_checkpoint,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            enable_key=enable_key,
            enable=enable,
            device=device,
            cls_ids=cls_ids,
            cls_names=cls_names,
            bbox_thr=bbox_thr)

        # item of _clip_buffer: (clip message, num of frames)
        self._clip_buffer = []
        self.score_thr = score_thr
        self.min_frame = min_frame
        self.fps = fps

    @property
    def totol_clip_length(self):
        return sum([clip[1] for clip in self._clip_buffer])

    def _add_clips(self, clips: List[Message]):
        """Push the newly loaded clips from buffer, and discard old clips."""
        for clip in clips:
            clip_length = clip.get_image().shape[0]
            self._clip_buffer.append((clip, clip_length))
        total_length = 0
        for i in range(-2, -len(self._clip_buffer) - 1, -1):
            total_length += self._clip_buffer[i][1]
            if total_length >= self.min_frame:
                self._clip_buffer = self._clip_buffer[i:]
                break

    def _merge_clips(self):
        """Concat the clips into a longer video, and gather bboxes."""
        videos = [clip[0].get_image() for clip in self._clip_buffer]
        video = np.concatenate(videos)
        det_results_lst = [
            clip[0].get_detection_results() for clip in self._clip_buffer
        ]
        bboxes = [
            self._process_clip_bbox(det_results)
            for det_results in det_results_lst
        ]
        bboxes = list(filter(len, bboxes))
        return video, bboxes

    def _process_clip_bbox(self, det_results: List[Dict]):
        """Filter and merge the bboxes of a video."""
        full_det_preds = []
        for det_result in det_results:
            det_preds = det_result['preds']
            if self.cls_ids:
                # Filter detection results by class ID
                det_preds = [
                    p for p in det_preds if p['cls_id'] in self.cls_ids
                ]
            elif self.cls_names:
                # Filter detection results by class name
                det_preds = [
                    p for p in det_preds if p['label'] in self.cls_names
                ]
            if self.bbox_thr > 0:
                det_preds = [
                    p for p in det_preds if p['bbox'][-1] > self.bbox_thr
                ]
            full_det_preds.extend(det_preds)

        merged_bbox = self._merge_bbox(full_det_preds)
        return merged_bbox

    def _merge_bbox(self, bboxes: List[Dict], ratio=0.5):
        """Merge bboxes in a video to create a new bbox that covers the region
        where hand moves in the video."""

        if len(bboxes) <= 1:
            return bboxes

        def compute_area(bbox):
            area = abs(bbox['bbox'][2] -
                       bbox['bbox'][0]) * abs(bbox['bbox'][3] -
                                              bbox['bbox'][1])
            return area

        bboxes.sort(key=lambda b: compute_area(b), reverse=True)
        merged = False
        for i in range(1, len(bboxes)):
            small_area = compute_area(bboxes[i])
            x1 = max(bboxes[0]['bbox'][0], bboxes[i]['bbox'][0])
            y1 = max(bboxes[0]['bbox'][1], bboxes[i]['bbox'][1])
            x2 = min(bboxes[0]['bbox'][2], bboxes[i]['bbox'][2])
            y2 = min(bboxes[0]['bbox'][3], bboxes[i]['bbox'][3])
            area_ratio = (abs(x2 - x1) * abs(y2 - y1)) / small_area
            if area_ratio > ratio:
                bboxes[0]['bbox'][0] = min(bboxes[0]['bbox'][0],
                                           bboxes[i]['bbox'][0])
                bboxes[0]['bbox'][1] = min(bboxes[0]['bbox'][1],
                                           bboxes[i]['bbox'][1])
                bboxes[0]['bbox'][2] = max(bboxes[0]['bbox'][2],
                                           bboxes[i]['bbox'][2])
                bboxes[0]['bbox'][3] = max(bboxes[0]['bbox'][3],
                                           bboxes[i]['bbox'][3])
                merged = True
                break

        if merged:
            bboxes.pop(i)
            return self._merge_bbox(bboxes, ratio)
        else:
            return [bboxes[0]]

    def process(self, input_msgs: Dict[str, Message]) -> Message:
        """Load and process the clips with hand detection result, and recognize
        the gesture."""

        input_msg = input_msgs['input']
        self._add_clips(input_msg)
        video, bboxes = self._merge_clips()
        msg = input_msg[-1]

        gesture_result = {
            'preds': [],
            'model_cfg': self.model.cfg.copy(),
        }
        if self.totol_clip_length >= self.min_frame and len(
                bboxes) > 0 and max(map(len, bboxes)) > 0:
            # Inference gesture
            pred_label, pred_score = inference_gesture_model(
                self.model,
                video,
                bboxes=bboxes,
                dataset_info=dict(
                    name='camera', fps=self.fps, modality=['rgb']))
            for i in range(len(pred_label)):
                result = bboxes[-1][0].copy()
                if pred_score[i] > self.score_thr:
                    label = pred_label[i].item()
                    label = self.model.cfg.dataset_info.category_info[label]
                    result['label'] = label
                gesture_result['preds'].append(result)

        msg.add_pose_result(gesture_result, tag=self.name)

        return msg
