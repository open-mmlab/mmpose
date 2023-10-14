# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.infer.infer import ModelType
from mmengine.logging import print_log
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmpose.evaluation.functional import nms
from mmpose.registry import INFERENCERS
from mmpose.structures import PoseDataSample, merge_data_samples
from .base_mmpose_inferencer import BaseMMPoseInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module()
class Hand3DInferencer(BaseMMPoseInferencer):
    """The inferencer for 3D hand pose estimation.

    Args:
        model (str, optional): Pretrained 2D pose estimation algorithm.
            It's the path to the config file or the model name defined in
            metafile. For example, it could be:

            - model alias, e.g. ``'body'``,
            - config name, e.g. ``'simcc_res50_8xb64-210e_coco-256x192'``,
            - config path

            Defaults to ``None``.
        weights (str, optional): Path to the checkpoint. If it is not
            specified and "model" is a model name of metafile, the weights
            will be loaded from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the
            available device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to "mmpose".
        det_model (str, optional): Config path or alias of detection model.
            Defaults to None.
        det_weights (str, optional): Path to the checkpoints of detection
            model. Defaults to None.
        det_cat_ids (int or list[int], optional): Category id for
            detection model. Defaults to None.
    """

    preprocess_kwargs: set = {'bbox_thr', 'nms_thr', 'bboxes'}
    forward_kwargs: set = {'disable_rebase_keypoint'}
    visualize_kwargs: set = {
        'return_vis',
        'show',
        'wait_time',
        'draw_bbox',
        'radius',
        'thickness',
        'kpt_thr',
        'vis_out_dir',
        'num_instances',
    }
    postprocess_kwargs: set = {'pred_out_dir', 'return_datasample'}

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmpose',
                 det_model: Optional[Union[ModelType, str]] = None,
                 det_weights: Optional[str] = None,
                 det_cat_ids: Optional[Union[int, Tuple]] = None) -> None:

        init_default_scope(scope)
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)
        self.model = revert_sync_batchnorm(self.model)

        # assign dataset metainfo to self.visualizer
        self.visualizer.set_dataset_meta(self.model.dataset_meta)

        # initialize hand detector
        self._init_detector(
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=det_cat_ids,
            device=device,
        )

        self._video_input = False
        self._buffer = defaultdict(list)

    def preprocess_single(self,
                          input: InputType,
                          index: int,
                          bbox_thr: float = 0.3,
                          nms_thr: float = 0.3,
                          bboxes: Union[List[List], List[np.ndarray],
                                        np.ndarray] = []):
        """Process a single input into a model-feedable format.

        Args:
            input (InputType): Input given by user.
            index (int): index of the input
            bbox_thr (float): threshold for bounding box detection.
                Defaults to 0.3.
            nms_thr (float): IoU threshold for bounding box NMS.
                Defaults to 0.3.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """

        if isinstance(input, str):
            data_info = dict(img_path=input)
        else:
            data_info = dict(img=input, img_path=f'{index}.jpg'.rjust(10, '0'))
        data_info.update(self.model.dataset_meta)

        if self.detector is not None:
            try:
                det_results = self.detector(
                    input, return_datasamples=True)['predictions']
            except ValueError:
                print_log(
                    'Support for mmpose and mmdet versions up to 3.1.0 '
                    'will be discontinued in upcoming releases. To '
                    'ensure ongoing compatibility, please upgrade to '
                    'mmdet version 3.2.0 or later.',
                    logger='current',
                    level=logging.WARNING)
                det_results = self.detector(
                    input, return_datasample=True)['predictions']
            pred_instance = det_results[0].pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)

            label_mask = np.zeros(len(bboxes), dtype=np.uint8)
            for cat_id in self.det_cat_ids:
                label_mask = np.logical_or(label_mask,
                                           pred_instance.labels == cat_id)

            bboxes = bboxes[np.logical_and(label_mask,
                                           pred_instance.scores > bbox_thr)]
            bboxes = bboxes[nms(bboxes, nms_thr)]

        data_infos = []
        if len(bboxes) > 0:
            for bbox in bboxes:
                inst = data_info.copy()
                inst['bbox'] = bbox[None, :4]
                inst['bbox_score'] = bbox[4:5]
                data_infos.append(self.pipeline(inst))
        else:
            inst = data_info.copy()

            # get bbox from the image size
            if isinstance(input, str):
                input = mmcv.imread(input)
            h, w = input.shape[:2]

            inst['bbox'] = np.array([[0, 0, w, h]], dtype=np.float32)
            inst['bbox_score'] = np.ones(1, dtype=np.float32)
            data_infos.append(self.pipeline(inst))

        return data_infos

    @torch.no_grad()
    def forward(self,
                inputs: Union[dict, tuple],
                disable_rebase_keypoint: bool = False):
        """Performs a forward pass through the model.

        Args:
            inputs (Union[dict, tuple]): The input data to be processed. Can
                be either a dictionary or a tuple.
            disable_rebase_keypoint (bool, optional): Flag to disable rebasing
                the height of the keypoints. Defaults to False.

        Returns:
            A list of data samples with prediction instances.
        """
        data_samples = self.model.test_step(inputs)
        data_samples_2d = []

        for idx, res in enumerate(data_samples):
            pred_instances = res.pred_instances
            keypoints = pred_instances.keypoints
            rel_root_depth = pred_instances.rel_root_depth
            scores = pred_instances.keypoint_scores
            hand_type = pred_instances.hand_type

            res_2d = PoseDataSample()
            gt_instances = res.gt_instances.clone()
            pred_instances = pred_instances.clone()
            res_2d.gt_instances = gt_instances
            res_2d.pred_instances = pred_instances

            # add relative root depth to left hand joints
            keypoints[:, 21:, 2] += rel_root_depth

            # set joint scores according to hand type
            scores[:, :21] *= hand_type[:, [0]]
            scores[:, 21:] *= hand_type[:, [1]]
            # normalize kpt score
            if scores.max() > 1:
                scores /= 255

            res_2d.pred_instances.set_field(keypoints[..., :2].copy(),
                                            'keypoints')

            # rotate the keypoint to make z-axis correspondent to height
            # for better visualization
            vis_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            keypoints[..., :3] = keypoints[..., :3] @ vis_R

            # rebase height (z-axis)
            if not disable_rebase_keypoint:
                valid = scores > 0
                keypoints[..., 2] -= np.min(
                    keypoints[valid, 2], axis=-1, keepdims=True)

            data_samples[idx].pred_instances.keypoints = keypoints
            data_samples[idx].pred_instances.keypoint_scores = scores
            data_samples_2d.append(res_2d)

        data_samples = [merge_data_samples(data_samples)]
        data_samples_2d = merge_data_samples(data_samples_2d)

        self._buffer['pose2d_results'] = data_samples_2d

        return data_samples

    def visualize(
        self,
        inputs: list,
        preds: List[PoseDataSample],
        return_vis: bool = False,
        show: bool = False,
        draw_bbox: bool = False,
        wait_time: float = 0,
        radius: int = 3,
        thickness: int = 1,
        kpt_thr: float = 0.3,
        num_instances: int = 1,
        vis_out_dir: str = '',
        window_name: str = '',
    ) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            return_vis (bool): Whether to return images with predicted results.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (ms). Defaults to 0
            draw_bbox (bool): Whether to draw the bounding boxes.
                Defaults to False
            radius (int): Keypoint radius for visualization. Defaults to 3
            thickness (int): Link thickness for visualization. Defaults to 1
            kpt_thr (float): The threshold to visualize the keypoints.
                Defaults to 0.3
            vis_out_dir (str, optional): Directory to save visualization
                results w/o predictions. If left as empty, no file will
                be saved. Defaults to ''.
            window_name (str, optional): Title of display window.
            window_close_event_handler (callable, optional):

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if (not return_vis) and (not show) and (not vis_out_dir):
            return

        if getattr(self, 'visualizer', None) is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        self.visualizer.radius = radius
        self.visualizer.line_width = thickness

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img = mmcv.imread(single_input, channel_order='rgb')
            elif isinstance(single_input, np.ndarray):
                img = mmcv.bgr2rgb(single_input)
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')
            img_name = os.path.basename(pred.metainfo['img_path'])

            # since visualization and inference utilize the same process,
            # the wait time is reduced when a video input is utilized,
            # thereby eliminating the issue of inference getting stuck.
            wait_time = 1e-5 if self._video_input else wait_time

            if num_instances < 0:
                num_instances = len(pred.pred_instances)

            visualization = self.visualizer.add_datasample(
                window_name,
                img,
                data_sample=pred,
                det_data_sample=self._buffer['pose2d_results'],
                draw_gt=False,
                draw_bbox=draw_bbox,
                show=show,
                wait_time=wait_time,
                convert_keypoint=False,
                axis_azimuth=-115,
                axis_limit=200,
                axis_elev=15,
                kpt_thr=kpt_thr,
                num_instances=num_instances)
            results.append(visualization)

            if vis_out_dir:
                self.save_visualization(
                    visualization,
                    vis_out_dir,
                    img_name=img_name,
                )

        if return_vis:
            return results
        else:
            return []
