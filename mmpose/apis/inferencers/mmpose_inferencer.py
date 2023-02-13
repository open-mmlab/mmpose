# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmengine.config import Config, ConfigDict
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample
from .pose2d_inferencer import Pose2DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class MMPoseInferencer(BaseInferencer):
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
            the selected rec model. If it is not specified and "pose2d" is a
            model name of metafile, the weights will be loaded from metafile.
            Defaults to None.
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
    postprocess_kwargs: set = {'pred_out_dir'}

    def __init__(self,
                 pose2d: Optional[str] = None,
                 pose2d_weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmpose',
                 instance_type: Optional[str] = None,
                 det_model: Optional[Union[ModelType, str]] = None,
                 det_weights: Optional[str] = None,
                 det_cat_ids: Optional[Union[int, List]] = None) -> None:

        if pose2d is None:
            raise ValueError('2d pose estimation algorithm should provided.')

        self.visualizer = None
        if pose2d is not None:
            self.pose2d_inferencer = Pose2DInferencer(pose2d, pose2d_weights,
                                                      device, scope,
                                                      instance_type, det_model,
                                                      det_weights, det_cat_ids)
            self.mode = 'pose2d'

    def _init_pipeline(self, cfg: ConfigType) -> None:
        pass

    def forward(self, inputs: InputType, batch_size: int,
                **forward_kwargs) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.
            batch_size (int): Batch size. Defaults to 1.

        Returns:
            Dict: The prediction results. Possibly with keys "pose2d".
        """
        result = {}
        if self.mode == 'pose2d':
            predictions = self.pose2d_inferencer(
                inputs,
                return_datasample=True,
                return_vis=False,
                batch_size=batch_size,
                **forward_kwargs)['predictions']
            result['pose2d'] = predictions

        return result

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

        ori_inputs = self._inputs_to_list(inputs)

        preds = self.forward(ori_inputs, batch_size, **preprocess_kwargs,
                             **forward_kwargs)

        visualization = self.visualize(ori_inputs, preds, **visualize_kwargs)
        results = self.postprocess(preds, visualization, return_datasample,
                                   **postprocess_kwargs)
        return results

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
                    if hasattr(self, 'pose2d_inferencer'):
                        self.pose2d_inferencer.video_input = True
                    # split video frames into a temperory folder
                    tmp_folder = tempfile.TemporaryDirectory()
                    video = mmcv.VideoReader(inputs)
                    self.pose2d_inferencer.video_info = dict(
                        fps=video.fps,
                        name=os.path.basename(inputs),
                        tmp_folder=tmp_folder)
                    video.cvt2frames(tmp_folder.name, show_progress=False)
                    frames = sorted(os.listdir(tmp_folder.name))
                    inputs = [os.path.join(tmp_folder.name, f) for f in frames]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

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

        if 'pose2d' in self.mode:
            return self.pose2d_inferencer.visualize(inputs, preds['pose2d'],
                                                    **kwargs)

    def postprocess(self,
                    preds: List[PoseDataSample],
                    visualization: List[np.ndarray],
                    return_datasample=False,
                    pred_out_dir: str = '',
                    **kwargs) -> dict:
        """Simply apply postprocess of pose2d_inferencer.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (np.ndarray): Visualized predictions.
            return_datasample (bool): Whether to return results as datasamples.
                Defaults to False.
            pred_out_dir (str): Directory to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.
        """
        result_dict = defaultdict(list)

        result_dict['visualization'] = visualization

        if 'pose2d' in self.mode:
            result_dict['predictions'] = self.pose2d_inferencer.postprocess(
                preds['pose2d'], visualization, return_datasample,
                pred_out_dir, **kwargs)['predictions']

        return result_dict
