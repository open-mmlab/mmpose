# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import tempfile
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
import torch.nn as nn
from mmdet.apis.det_inferencer import DetInferencer
from mmengine.config import Config, ConfigDict
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmengine.structures import InstanceData
from rich.progress import track

from mmpose.apis.inference import dataset_meta_from_config
from mmpose.evaluation.functional import nms
from mmpose.registry import DATASETS, INFERENCERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from .utils import default_det_models

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='pose-estimation')
@INFERENCERS.register_module()
class Pose2DInferencer(BaseInferencer):
    """The inferencer for 2D pose estimation.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "td-hm_hrnet-w32_8xb64-210e_coco-256x192" or
            "configs/body_2d_keypoint/topdown_heatmap/coco/" \\
            "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
        weights (str, optional): Path to the checkpoint. If it is not
            specified and model is a model name of metafile, the weights
            will be loaded from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the
            available device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to "mmpose".
        instance_type (str, optional): The name of the instances, such as
            "human", "hand", "animal", and etc. This argument works as the
            alias for the detection models, which will be utilized in
            topdown methods. Defaults to None.
        det_model(str, optional): Path to the config of detection model.
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
        'radius',
        'thickness',
        'kpt_thr',
        'vis_out_dir',
    }
    postprocess_kwargs: set = {
        'pred_out_dir',
        'return_datasample',
    }

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmpose',
                 instance_type: Optional[str] = None,
                 det_model: Optional[Union[ModelType, str]] = None,
                 det_weights: Optional[str] = None,
                 det_cat_ids: Optional[Union[int, List]] = None) -> None:

        init_default_scope(scope)
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

        # assign dataset metainfo to self.visualizer
        self.visualizer.set_dataset_meta(self.model.dataset_meta)

        # initialize detector for top-down models
        if self.cfg.data_mode == 'topdown':
            if det_model is None:
                if instance_type is None:
                    instance_type = DATASETS.get(
                        self.cfg.dataset_type).__module__.split(
                            'datasets.')[-1].split('.')[0]
                if instance_type not in default_det_models:
                    raise ValueError(
                        f'detector for {instance_type} has not been '
                        'provided. You need to specify `det_model`, '
                        '`det_weights` and `det_cat_ids` manually')
                det_info = default_det_models[instance_type.lower()]
                det_model, det_weights, det_cat_ids = det_info[
                    'model'], det_info['weights'], det_info['cat_ids']

            self.detector = DetInferencer(
                det_model, det_weights, device=device)
            self.det_cat_ids = det_cat_ids

        self.video_input = False

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:
        """Loading model weights and meta information from cfg and checkpoint.

        Subclasses could override this method to load extra meta information
        from ``checkpoint`` and ``cfg`` to model.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """
        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get('meta', {})
            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint_meta:
                # mmpose 1.x
                model.dataset_meta = checkpoint_meta['dataset_meta']
            else:
                warnings.warn(
                    'dataset_meta are not saved in the checkpoint\'s '
                    'meta data, load via config.')
                model.dataset_meta = dataset_meta_from_config(
                    cfg, dataset_mode='train')
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
            model.dataset_meta = dataset_meta_from_config(
                cfg, dataset_mode='train')

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string
              according to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the
                # inputs as a directory
                filepath_list = [
                    join_path(inputs, fname)
                    for fname in list_dir_or_file(inputs, list_dir=False)
                ]
                inputs = []
                for filepath in filepath_list:
                    input_type = mimetypes.guess_type(filepath)[0].split(
                        '/')[0]
                    if input_type == 'image':
                        inputs.append(filepath)
                inputs = inputs
            else:
                # if inputs is a path to a video file, it will be converted
                # to a list containing separated frame filenames
                input_type = mimetypes.guess_type(inputs)[0].split('/')[0]
                if input_type == 'video':
                    self.video_input = True
                    # split video frames into a temperory folder
                    tmp_folder = tempfile.TemporaryDirectory()
                    video = mmcv.VideoReader(inputs)
                    self.video_info = dict(
                        fps=video.fps,
                        name=os.path.basename(inputs),
                        tmp_folder=tmp_folder)
                    video.cvt2frames(tmp_folder.name, show_progress=False)
                    frames = sorted(os.listdir(tmp_folder.name))
                    inputs = [os.path.join(tmp_folder.name, f) for f in frames]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _init_pipeline(self, cfg: ConfigType) -> Callable:
        """Initialize the test pipeline.

        Args:
            cfg (ConfigType): model config path or dict

        Returns:
            A pipeline to handle various input data, such as ``str``,
            ``np.ndarray``. The returned pipeline will be used to process
            a single data.
        """
        return Compose(cfg.test_dataloader.dataset.pipeline)

    def preprocess(self,
                   inputs: InputsType,
                   batch_size: int = 1,
                   bbox_thr: float = 0.3,
                   nms_thr: float = 0.3):
        """Process the inputs into a model-feedable format.

        1. for topdown models, detection is conducted in this function
        2. inputs are converted into the format same as the data_info
           provided by dataset and then fed to pipeline
        3. batch_size is ineffective here
        """

        for i, input in enumerate(inputs):

            if isinstance(input, str):
                data_info = dict(img_path=input)
            else:
                data_info = dict(img=input, img_path=f'{i+1}.jpg'.rjust(10))
            data_info.update(self.model.dataset_meta)

            if self.cfg.data_mode == 'topdown':
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

            yield self.collate_fn(data_infos)

    def forward(self, inputs: Union[dict, tuple]):
        data_samples = super().forward(inputs)
        if self.cfg.data_mode == 'topdown':
            data_samples = [merge_data_samples(data_samples)]
        return data_samples

    def __call__(
        self,
        inputs: InputsType,
        return_datasamples: bool = False,
        batch_size: int = 1,
        out_dir: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            vis_out_dir (str, optional): directory to save visualization
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
                kwargs['vis_out_dir'] = out_dir
            if 'pred_out_dir' not in kwargs:
                kwargs['pred_out_dir'] = out_dir

        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)
        preds = []
        if not hasattr(self, 'detector'):
            inputs = track(inputs, description='Inference')
        for data in inputs:
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(
            ori_inputs, preds,
            **visualize_kwargs)  # type: ignore  # noqa: E501
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results

    def visualize(self,
                  inputs: list,
                  preds: List[PoseDataSample],
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  radius: int = 3,
                  thickness: int = 1,
                  kpt_thr: float = 0.3,
                  vis_out_dir: Optional[str] = None) -> List[np.ndarray]:
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
                results. Defaults to None.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if (not return_vis) and (not show) and (vis_out_dir is None):
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
                img = single_input.copy()
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            img_name = os.path.basename(pred.metainfo['img_path'])

            if self.video_input:
                out_file = os.path.join(self.video_info['tmp_folder'].name,
                                        img_name)
            elif vis_out_dir:
                out_file = os.path.join(vis_out_dir, img_name)
            else:
                out_file = None

            visualization = self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                draw_gt=False,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                kpt_score_thr=kpt_thr)
            results.append(visualization)

        if self.video_input and vis_out_dir:
            # merge saved frame images into video
            mmcv.frames2video(
                self.video_info['tmp_folder'].name,
                f'{vis_out_dir}/{self.video_info["name"]}',
                fps=self.video_info['fps'],
                fourcc='mp4v',
                show_progress=False)

        return results

    def postprocess(
        self,
        preds: List[PoseDataSample],
        visualization: List[np.ndarray],
        return_datasample=False,
        pred_out_dir: str = '',
    ) -> dict:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (np.ndarray): Visualized predictions.
            return_datasample (bool): Whether to return results as
                datasamples. Defaults to False.
            pred_out_dir (str): Directory to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``

            - ``visualization (Any)``: Returned by :meth:`visualize`
            - ``predictions`` (dict or DataSample): Returned by
              :meth:`forward` and processed in :meth:`postprocess`.
              If ``return_datasample=False``, it usually should be a
              json-serializable dict containing only basic data elements such
              as strings and numbers.
        """

        result_dict = defaultdict(list)

        result_dict['visualization'] = visualization
        for pred in preds:
            if not return_datasample:
                # convert datasamples to list of instance predictions
                pred = split_instances(pred.pred_instances)
            result_dict['predictions'].append(pred)

        if self.video_input:
            result_dict['predictions'] = [{
                'frame_id': i,
                'instances': pred
            } for i, pred in enumerate(result_dict['predictions'])]

        if pred_out_dir != '':
            if self.video_input:
                fname = os.path.splitext(self.video_info['name'])[0] + '.json'
                mmengine.dump(result_dict['predictions'],
                              join_path(pred_out_dir, fname))
            else:
                for pred, data_sample in zip(result_dict['predictions'],
                                             preds):
                    fname = os.path.splitext(
                        os.path.basename(
                            data_sample.metainfo['img_path']))[0] + '.json'
                    mmengine.dump(pred, join_path(pred_out_dir, fname))

        return result_dict
