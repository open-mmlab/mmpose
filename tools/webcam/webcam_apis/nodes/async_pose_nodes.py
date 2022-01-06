# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import AsyncResult
from typing import Callable, Dict, List, Optional, Tuple, Union

from tools.webcam.webcam_apis.nodes.async_node import AsyncNode
from tools.webcam.webcam_apis.utils.message import Message

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model)
from .builder import NODES

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


@dataclass
class TrackInfo:
    """Dataclass for pose tracking information."""
    next_id: int = 0
    last_pose_preds: Optional[List] = None
    last_time: Optional[float] = None


@NODES.register_module()
class AsyncTwoStageKeypointDetectorNode(AsyncNode):

    def __init__(self,
                 name: str,
                 det_config: str,
                 det_checkpoint: str,
                 pose_config: str,
                 pose_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 cls_ids: Optional[List] = None,
                 cls_names: Optional[List] = None,
                 bbox_thr: float = 0.5,
                 num_workers: Optional[int] = None,
                 worker_timeout: Optional[float] = None,
                 backend: str = 'thread'):
        # Check mmdetection is installed
        assert has_mmdet, 'Please install mmdet to run the demo.'

        super().__init__(
            name=name,
            enable_key=enable_key,
            enable=enable,
            num_workers=num_workers,
            worker_timeout=worker_timeout,
            backend=backend)

        self.det_config = det_config
        self.det_checkpoint = det_checkpoint
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.cls_ids = cls_ids
        self.cls_names = cls_names
        self.bbox_thr = bbox_thr

        # Store history for pose tracking
        self.track_info = TrackInfo(0, [], None)

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', essential=True)
        self.register_output_buffer(output_buffer)

    def get_pool_initializer(self) -> Callable:
        return _init_global_models

    def get_pool_initargs(self) -> Tuple:
        return (self.det_config, self.det_checkpoint, self.pose_config,
                self.pose_checkpoint)

    def get_pool_bypass_func(self) -> Callable:
        return lambda msgs: msgs['input']

    def get_pool_process_func(self) -> Callable:
        return partial(
            _inference_models,
            input_name='input',
            cls_ids=self.cls_ids,
            cls_names=self.cls_names,
            bbox_thr=self.bbox_thr,
            track_info=self.track_info,
            result_tag=self.name)

    def process_pool_output(self, res: AsyncResult) -> Optional[Message]:
        """Override process_pool_output() to process pose prediction and pose
        tracking status respectively."""
        output = super().process_pool_output(res)
        if output is None:
            return None
        output_msg, track_info = output
        self.track_info = track_info
        return output_msg


def _init_global_models(det_config, det_checkpoint, pose_config,
                        pose_checkpoint):
    global det_model, pose_model
    det_model = init_detector(det_config, det_checkpoint, device='cpu')
    pose_model = init_pose_model(pose_config, pose_checkpoint, device='cpu')


def _inference_models(input_msgs: Dict[str, Message],
                      input_name: str,
                      cls_ids: Optional[List] = None,
                      cls_names: Optional[List] = None,
                      bbox_thr: float = 0.5,
                      track_info: Optional[TrackInfo] = None,
                      result_tag: Optional[str] = None
                      ) -> Tuple[Message, Optional[Dict]]:
    global det_model, pose_model

    input_msg = input_msgs[input_name]
    img = input_msg.get_image()

    # Perform object detection
    preds = inference_detector(det_model, img)

    # Postprocess detection result
    if isinstance(preds, tuple):
        preds = preds[0]
    assert len(preds) == len(det_model.CLASSES)

    det_result = {'preds': [], 'model_cfg': det_model.cfg.copy()}
    for i, (cls_name, bboxes) in enumerate(zip(det_model.CLASSES, preds)):
        preds_i = [{
            'cls_id': i,
            'label': cls_name,
            'bbox': bbox
        } for bbox in bboxes]
        det_result['preds'].extend(preds_i)

    input_msg.add_detection_result(det_result, tag=result_tag)

    if cls_ids:
        cls_det_preds = [
            p for p in det_result['preds'] if p['cls_id'] in cls_ids
        ]
    elif cls_name:
        cls_det_preds = [
            p for p in det_result['preds'] if p['label'] in cls_names
        ]

    # Inference pose
    pose_preds, _ = inference_top_down_pose_model(
        pose_model, img, cls_det_preds, bbox_thr=bbox_thr, format='xyxy')

    # Pose tracking
    if track_info:
        current_time = time.time()
        if track_info.last_time is None:
            fps = None
        else:
            fps = 1.0 / (current_time - track_info.last_time)

        pose_preds, next_id = get_track_id(
            pose_preds,
            track_info.last_pose_preds,
            track_info.next_id,
            use_oks=False,
            tracking_thr=0.3,
            use_one_euro=True,
            fps=fps)

        track_info = TrackInfo(next_id, pose_preds.copy(), current_time)

    pose_result = {
        'preds': pose_preds,
        'model_cfg': pose_model.cfg.copy(),
    }

    input_msg.add_pose_result(pose_result, tag=result_tag)

    return input_msg, track_info
