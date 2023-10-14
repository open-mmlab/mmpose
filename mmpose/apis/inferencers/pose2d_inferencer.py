# Copyright (c) OpenMMLab. All rights reserved.
import logging
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
from mmpose.structures import merge_data_samples
from .base_mmpose_inferencer import BaseMMPoseInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='pose-estimation')
@INFERENCERS.register_module()
class Pose2DInferencer(BaseMMPoseInferencer):
    """The inferencer for 2D pose estimation.

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
    forward_kwargs: set = {'merge_results'}
    visualize_kwargs: set = {
        'return_vis',
        'show',
        'wait_time',
        'draw_bbox',
        'radius',
        'thickness',
        'kpt_thr',
        'vis_out_dir',
        'skeleton_style',
        'draw_heatmap',
        'black_background',
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

        # initialize detector for top-down models
        if self.cfg.data_mode == 'topdown':
            self._init_detector(
                det_model=det_model,
                det_weights=det_weights,
                det_cat_ids=det_cat_ids,
                device=device,
            )

        self._video_input = False

    def update_model_visualizer_settings(self,
                                         draw_heatmap: bool = False,
                                         skeleton_style: str = 'mmpose',
                                         **kwargs) -> None:
        """Update the settings of models and visualizer according to inference
        arguments.

        Args:
            draw_heatmaps (bool, optional): Flag to visualize predicted
                heatmaps. If not provided, it defaults to False.
            skeleton_style (str, optional): Skeleton style selection. Valid
                options are 'mmpose' and 'openpose'. Defaults to 'mmpose'.
        """
        self.model.test_cfg['output_heatmaps'] = draw_heatmap

        if skeleton_style not in ['mmpose', 'openpose']:
            raise ValueError('`skeleton_style` must be either \'mmpose\' '
                             'or \'openpose\'')

        if skeleton_style == 'openpose':
            self.visualizer.set_dataset_meta(self.model.dataset_meta,
                                             skeleton_style)

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

        if self.cfg.data_mode == 'topdown':
            bboxes = []
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
                    (pred_instance.bboxes, pred_instance.scores[:, None]),
                    axis=1)

                label_mask = np.zeros(len(bboxes), dtype=np.uint8)
                for cat_id in self.det_cat_ids:
                    label_mask = np.logical_or(label_mask,
                                               pred_instance.labels == cat_id)

                bboxes = bboxes[np.logical_and(
                    label_mask, pred_instance.scores > bbox_thr)]
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

        else:  # bottom-up
            data_infos = [self.pipeline(data_info)]

        return data_infos

    @torch.no_grad()
    def forward(self,
                inputs: Union[dict, tuple],
                merge_results: bool = True,
                bbox_thr: float = -1):
        """Performs a forward pass through the model.

        Args:
            inputs (Union[dict, tuple]): The input data to be processed. Can
                be either a dictionary or a tuple.
            merge_results (bool, optional): Whether to merge data samples,
                default to True. This is only applicable when the data_mode
                is 'topdown'.
            bbox_thr (float, optional): A threshold for the bounding box
                scores. Bounding boxes with scores greater than this value
                will be retained. Default value is -1 which retains all
                bounding boxes.

        Returns:
            A list of data samples with prediction instances.
        """
        data_samples = self.model.test_step(inputs)
        if self.cfg.data_mode == 'topdown' and merge_results:
            data_samples = [merge_data_samples(data_samples)]
        if bbox_thr > 0:
            for ds in data_samples:
                if 'bbox_scores' in ds.pred_instances:
                    ds.pred_instances = ds.pred_instances[
                        ds.pred_instances.bbox_scores > bbox_thr]
        return data_samples
