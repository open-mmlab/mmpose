# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.infer.infer import ModelType
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from rich.progress import track

from mmpose.evaluation.functional import nms
from mmpose.registry import DATASETS, INFERENCERS
from mmpose.structures import merge_data_samples
from .base_mmpose_inferencer import BaseMMPoseInferencer
from .utils import default_det_models

try:
    from mmdet.apis.det_inferencer import DetInferencer
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

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
        det_model(str, optional): Config path or alias of detection model.
            Defaults to None.
        det_weights(str, optional): Path to the checkpoints of detection
            model. Defaults to None.
        det_cat_ids(int or list[int], optional): Category id for
            detection model. Defaults to None.
    """

    preprocess_kwargs: set = {'bbox_thr', 'nms_thr'}
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis',
        'show',
        'wait_time',
        'draw_bbox',
        'radius',
        'thickness',
        'kpt_thr',
        'vis_out_dir',
    }
    postprocess_kwargs: set = {'pred_out_dir'}

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
            det_scope = 'mmdet'
            if det_model is None:
                det_model = DATASETS.get(
                    self.cfg.dataset_type).__module__.split(
                        'datasets.')[-1].split('.')[0].lower()
                det_info = default_det_models[det_model]
                det_model, det_weights, det_cat_ids = det_info[
                    'model'], det_info['weights'], det_info['cat_ids']
            elif os.path.exists(det_model):
                det_cfg = Config.fromfile(det_model)
                det_scope = det_cfg.default_scope

            if has_mmdet:
                self.detector = DetInferencer(
                    det_model, det_weights, device=device, scope=det_scope)
                self.detector.model = revert_sync_batchnorm(
                    self.detector.model)
            else:
                raise RuntimeError(
                    'MMDetection (v3.0.0rc6 or above) is required to '
                    'build inferencers for top-down pose estimation models.')

            if isinstance(det_cat_ids, (tuple, list)):
                self.det_cat_ids = det_cat_ids
            else:
                self.det_cat_ids = (det_cat_ids, )

        self._video_input = False

    def preprocess_single(self,
                          input: InputType,
                          index: int,
                          bbox_thr: float = 0.3,
                          nms_thr: float = 0.3):
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

        else:  # bottom-up
            data_infos = [self.pipeline(data_info)]

        return data_infos

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], bbox_thr=-1):
        data_samples = super().forward(inputs)
        if self.cfg.data_mode == 'topdown':
            data_samples = [merge_data_samples(data_samples)]
        if bbox_thr > 0:
            for ds in data_samples:
                if 'bbox_scores' in ds.pred_instances:
                    ds.pred_instances = ds.pred_instances[
                        ds.pred_instances.bbox_scores > bbox_thr]
        return data_samples

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

        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        # preprocessing
        if isinstance(inputs, str) and inputs.startswith('webcam'):
            inputs = self._get_webcam_inputs(inputs)
            batch_size = 1
            if not visualize_kwargs.get('show', False):
                warnings.warn('The display mode is closed when using webcam '
                              'input. It will be turned on automatically.')
            visualize_kwargs['show'] = True
        else:
            inputs = self._inputs_to_list(inputs)

        forward_kwargs['bbox_thr'] = preprocess_kwargs.get('bbox_thr', -1)
        inputs = self.preprocess(
            inputs, batch_size=batch_size, **preprocess_kwargs)

        preds = []
        if not hasattr(self, 'detector'):
            inputs = track(inputs, description='Inference')

        for proc_inputs, ori_inputs in inputs:
            preds = self.forward(proc_inputs, **forward_kwargs)

            visualization = self.visualize(ori_inputs, preds,
                                           **visualize_kwargs)
            results = self.postprocess(preds, visualization, return_datasample,
                                       **postprocess_kwargs)
            yield results

        # merge visualization and prediction results
        if self._video_input:
            self._merge_outputs(**visualize_kwargs, **postprocess_kwargs)
