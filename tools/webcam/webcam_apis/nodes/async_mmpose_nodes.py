# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import AsyncResult
from typing import Callable, Dict, List, Optional, Tuple, Union

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model)
from ..utils import Message
from .async_node import AsyncNode
from .builder import NODES


@dataclass
class TrackInfo:
    """Dataclass for pose tracking information."""
    next_id: int = 0
    last_pose_preds: Optional[List] = None
    last_time: Optional[float] = None


@NODES.register_module()
class AsyncCPUTopDownPoseEstimatorNode(AsyncNode):

    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 cls_ids: Optional[List] = None,
                 cls_names: Optional[List] = None,
                 bbox_thr: float = 0.5,
                 num_workers: Optional[int] = None,
                 worker_timeout: Optional[float] = None):
        super().__init__(
            name=name,
            enable_key=enable_key,
            enable=enable,
            num_workers=num_workers,
            worker_timeout=worker_timeout)

        # Init model
        self.model_config = model_config
        self.model_checkpoint = model_checkpoint

        self.cls_ids = cls_ids
        self.cls_names = cls_names
        self.bbox_thr = bbox_thr

        # Store history for pose tracking
        self.track_info = TrackInfo(0, [], None)

        # Register buffers
        self.register_input_buffer(input_buffer, 'input', essential=True)
        self.register_output_buffer(output_buffer)

    def get_pool_initializer(self) -> Callable:
        return _init_global_model

    def get_pool_initargs(self) -> Tuple:
        return self.model_config, self.model_checkpoint

    def get_pool_bypass_func(self) -> Callable:
        return lambda msgs: msgs['input']

    def get_pool_process_func(self) -> Callable:
        return partial(
            _inference_pose_estimator,
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


def _init_global_model(config, checkpoint):
    """A helper function to init a global pose model.

    This function is for initializing a worker in a multiprocessing.pool.Pool.
    """
    global model
    model = init_pose_model(config, checkpoint, device='cpu')


def _inference_pose_estimator(input_msgs: Dict[str, Message],
                              input_name: str,
                              cls_ids: Optional[List] = None,
                              cls_names: Optional[List] = None,
                              bbox_thr: float = 0.5,
                              track_info: Optional[TrackInfo] = None,
                              result_tag: Optional[str] = None
                              ) -> Tuple[Message, Optional[Dict]]:
    """A helper function to perform pose estimation.

    This function is for inference a global pose model on an input frame image
    in a multiprocessing.pool.Pool.
    """

    global model

    input_msg = input_msgs[input_name]
    img = input_msg.get_image()
    det_results = input_msg.get_detection_results()

    if det_results is None:
        raise ValueError(
            'No detection results are found in the frame message.'
            'The pose estimator node should be placed after a detector node.')

    full_det_preds = []
    for det_result in det_results:
        det_preds = det_result['preds']
        if cls_ids:
            # Filter detection results by class ID
            det_preds = [p for p in det_preds if p['cls_id'] in cls_ids]
        elif cls_names:
            # Filter detection results by class name
            det_preds = [p for p in det_preds if p['label'] in cls_names]
        full_det_preds.extend(det_preds)

    # Inference pose
    pose_preds, _ = inference_top_down_pose_model(
        model, img, full_det_preds, bbox_thr=bbox_thr, format='xyxy')

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
        'model_cfg': model.cfg.copy(),
    }

    input_msg.add_pose_result(pose_result, tag=result_tag)

    return input_msg, track_info
