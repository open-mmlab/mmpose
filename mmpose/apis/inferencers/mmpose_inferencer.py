# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData

from .base_mmpose_inferencer import BaseMMPoseInferencer
from .pose2d_inferencer import Pose2DInferencer
from .pose3d_inferencer import Pose3DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class MMPoseInferencer(BaseMMPoseInferencer):
    """MMPose Inferencer. It's a unified inferencer interface for pose
    estimation task, currently including: Pose2D. and it can be used to perform
    2D keypoint detection.

    Args:
        pose2d (str, optional): Pretrained 2D pose estimation algorithm.
            It's the path to the config file or the model name defined in
            metafile. For example, it could be:

            - model alias, e.g. ``'body'``,
            - config name, e.g. ``'simcc_res50_8xb64-210e_coco-256x192'``,
            - config path

            Defaults to ``None``.
        pose2d_weights (str, optional): Path to the custom checkpoint file of
            the selected pose2d model. If it is not specified and "pose2d" is
            a model name of metafile, the weights will be loaded from
            metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the
            available device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to "mmpose".
        det_model(str, optional): Config path or alias of detection model.
            Defaults to None.
        det_weights(str, optional): Path to the checkpoints of detection
            model. Defaults to None.
        det_cat_ids(int or list[int], optional): Category id for
            detection model. Defaults to None.
        output_heatmaps (bool, optional): Flag to visualize predicted
            heatmaps. If set to None, the default setting from the model
            config will be used. Default is None.
    """

    preprocess_kwargs: set = {
        'bbox_thr', 'nms_thr', 'bboxes', 'use_oks_tracking', 'tracking_thr',
        'norm_pose_2d'
    }
    forward_kwargs: set = {'rebase_keypoint_height'}
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_bbox', 'radius', 'thickness',
        'kpt_thr', 'vis_out_dir', 'skeleton_style', 'draw_heatmap',
        'black_background'
    }
    postprocess_kwargs: set = {'pred_out_dir'}

    def __init__(self,
                 pose2d: Optional[str] = None,
                 pose2d_weights: Optional[str] = None,
                 pose3d: Optional[str] = None,
                 pose3d_weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmpose',
                 det_model: Optional[Union[ModelType, str]] = None,
                 det_weights: Optional[str] = None,
                 det_cat_ids: Optional[Union[int, List]] = None) -> None:

        self.visualizer = None
        if pose3d is not None:
            self.inferencer = Pose3DInferencer(pose3d, pose3d_weights, pose2d,
                                               pose2d_weights, device, scope,
                                               det_model, det_weights,
                                               det_cat_ids)
        elif pose2d is not None:
            self.inferencer = Pose2DInferencer(pose2d, pose2d_weights, device,
                                               scope, det_model, det_weights,
                                               det_cat_ids)
        else:
            raise ValueError('Either 2d or 3d pose estimation algorithm '
                             'should be provided.')

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
            List[str or np.ndarray]: List of original inputs in the batch
        """

        for i, input in enumerate(inputs):
            data_batch = {}
            data_infos = self.inferencer.preprocess_single(
                input, index=i, **kwargs)
            data_batch = self.inferencer.collate_fn(data_infos)
            # only supports inference with batch size 1
            yield data_batch, [input]

    @torch.no_grad()
    def forward(self, inputs: InputType, **forward_kwargs) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.

        Returns:
            Dict: The prediction results. Possibly with keys "pose2d".
        """
        return self.inferencer.forward(inputs, **forward_kwargs)

    def __call__(
        self,
        inputs: InputsType,
        return_datasample: bool = False,
        batch_size: int = 1,
        out_dir: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasample (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            out_dir (str, optional): directory to save visualization
                results and predictions. Will be overoden if vis_out_dir or
                pred_out_dir are given. Defaults to None
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``,
                ``visualize_kwargs`` and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        if out_dir is not None:
            if 'vis_out_dir' not in kwargs:
                kwargs['vis_out_dir'] = f'{out_dir}/visualizations'
            if 'pred_out_dir' not in kwargs:
                kwargs['pred_out_dir'] = f'{out_dir}/predictions'

        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in set.union(self.inferencer.preprocess_kwargs,
                                self.inferencer.forward_kwargs,
                                self.inferencer.visualize_kwargs,
                                self.inferencer.postprocess_kwargs)
        }
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        self.inferencer.update_model_visualizer_settings(**kwargs)

        # preprocessing
        if isinstance(inputs, str) and inputs.startswith('webcam'):
            inputs = self.inferencer._get_webcam_inputs(inputs)
            batch_size = 1
            if not visualize_kwargs.get('show', False):
                warnings.warn('The display mode is closed when using webcam '
                              'input. It will be turned on automatically.')
            visualize_kwargs['show'] = True
        else:
            inputs = self.inferencer._inputs_to_list(inputs)
        self._video_input = self.inferencer._video_input
        if self._video_input:
            self.video_info = self.inferencer.video_info

        inputs = self.preprocess(
            inputs, batch_size=batch_size, **preprocess_kwargs)

        # forward
        if 'bbox_thr' in self.inferencer.forward_kwargs:
            forward_kwargs['bbox_thr'] = preprocess_kwargs.get('bbox_thr', -1)

        preds = []

        for proc_inputs, ori_inputs in inputs:
            preds = self.forward(proc_inputs, **forward_kwargs)

            visualization = self.visualize(ori_inputs, preds,
                                           **visualize_kwargs)
            results = self.postprocess(preds, visualization, return_datasample,
                                       **postprocess_kwargs)
            yield results

        if self._video_input:
            self._finalize_video_processing(
                postprocess_kwargs.get('pred_out_dir', ''))

    def visualize(self, inputs: InputsType, preds: PredType,
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            return_vis (bool): Whether to return images with predicted results.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            show_interval (int): The interval of show (s). Defaults to 0
            radius (int): Keypoint radius for visualization. Defaults to 3
            thickness (int): Link thickness for visualization. Defaults to 1
            kpt_thr (float): The threshold to visualize the keypoints.
                Defaults to 0.3
            vis_out_dir (str, optional): directory to save visualization
                results w/o predictions. If left as empty, no file will
                be saved. Defaults to ''.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        window_name = ''
        if self.inferencer._video_input:
            window_name = self.inferencer.video_info['name']

        return self.inferencer.visualize(
            inputs, preds, window_name=window_name, **kwargs)
