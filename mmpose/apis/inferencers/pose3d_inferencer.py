# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.fileio import join_path
from mmengine.infer.infer import ModelType
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmengine.utils import mkdir_or_exist

from mmpose.apis import (_track_by_iou, _track_by_oks, collate_pose_sequence,
                         convert_keypoint_definition, extract_pose_sequence)
from mmpose.registry import INFERENCERS
from mmpose.structures import PoseDataSample, merge_data_samples
from .base_mmpose_inferencer import BaseMMPoseInferencer
from .pose2d_inferencer import Pose2DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='pose-estimation-3d')
@INFERENCERS.register_module()
class Pose3DInferencer(BaseMMPoseInferencer):
    """The inferencer for 3D pose estimation.

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
                 pose2d_model: Optional[Union[ModelType, str]] = None,
                 pose2d_weights: Optional[str] = None,
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

        # initialize 2d pose estimator
        self.pose2d_model = Pose2DInferencer(
            pose2d_model if pose2d_model else 'human', pose2d_weights, device,
            scope, det_model, det_weights, det_cat_ids)

        # helper functions
        self._keypoint_converter = partial(
            convert_keypoint_definition,
            pose_det_dataset=self.pose2d_model.cfg.test_dataloader.
            dataset['type'],
            pose_lift_dataset=self.cfg.test_dataloader.dataset['type'],
        )

        self._pose_seq_extractor = partial(
            extract_pose_sequence,
            causal=self.cfg.test_dataloader.dataset.get('causal', False),
            seq_len=self.cfg.test_dataloader.dataset.get('seq_len', 1),
            step=self.cfg.test_dataloader.dataset.get('seq_step', 1))

        self._video_input = False
        self._buffer = defaultdict(list)

    def preprocess_single(self,
                          input: InputType,
                          index: int,
                          bbox_thr: float = 0.3,
                          nms_thr: float = 0.3,
                          bboxes: Union[List[List], List[np.ndarray],
                                        np.ndarray] = [],
                          use_oks_tracking: bool = False,
                          tracking_thr: float = 0.3,
                          norm_pose_2d: bool = False):
        """Process a single input into a model-feedable format.

        Args:
            input (InputType): The input provided by the user.
            index (int): The index of the input.
            bbox_thr (float, optional): The threshold for bounding box
                detection. Defaults to 0.3.
            nms_thr (float, optional): The Intersection over Union (IoU)
                threshold for bounding box Non-Maximum Suppression (NMS).
                Defaults to 0.3.
            bboxes (Union[List[List], List[np.ndarray], np.ndarray]):
                The bounding boxes to use. Defaults to [].
            use_oks_tracking (bool, optional): A flag that indicates
                whether OKS-based tracking should be used. Defaults to False.
            tracking_thr (float, optional): The threshold for tracking.
                Defaults to 0.3.
            norm_pose_2d (bool, optional): A flag that indicates whether 2D
                pose normalization should be used. Defaults to False.

        Yields:
            Any: The data processed by the pipeline and collate_fn.

        This method first calculates 2D keypoints using the provided
        pose2d_model. The method also performs instance matching, which
        can use either OKS-based tracking or IOU-based tracking.
        """

        # calculate 2d keypoints
        results_pose2d = next(
            self.pose2d_model(
                input,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                bboxes=bboxes,
                merge_results=False,
                return_datasample=True))['predictions']

        for ds in results_pose2d:
            ds.pred_instances.set_field(
                (ds.pred_instances.bboxes[..., 2:] -
                 ds.pred_instances.bboxes[..., :2]).prod(-1), 'areas')

        if not self._video_input:
            height, width = results_pose2d[0].metainfo['ori_shape']

            # Clear the buffer if inputs are individual images to prevent
            # carryover effects from previous images
            self._buffer.clear()

        else:
            height = self.video_info['height']
            width = self.video_info['width']
        img_path = results_pose2d[0].metainfo['img_path']

        # instance matching
        if use_oks_tracking:
            _track = partial(_track_by_oks)
        else:
            _track = _track_by_iou

        for result in results_pose2d:
            track_id, self._buffer['results_pose2d_last'], _ = _track(
                result, self._buffer['results_pose2d_last'], tracking_thr)
            if track_id == -1:
                pred_instances = result.pred_instances.cpu().numpy()
                keypoints = pred_instances.keypoints
                if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    next_id = self._buffer.get('next_id', 0)
                    result.set_field(next_id, 'track_id')
                    self._buffer['next_id'] = next_id + 1
                else:
                    # If the number of keypoints detected is small,
                    # delete that person instance.
                    result.pred_instances.keypoints[..., 1] = -10
                    result.pred_instances.bboxes *= 0
                    result.set_field(-1, 'track_id')
            else:
                result.set_field(track_id, 'track_id')
        self._buffer['pose2d_results'] = merge_data_samples(results_pose2d)

        # convert keypoints
        results_pose2d_converted = [ds.cpu().numpy() for ds in results_pose2d]
        for ds in results_pose2d_converted:
            ds.pred_instances.keypoints = self._keypoint_converter(
                ds.pred_instances.keypoints)
        self._buffer['pose_est_results_list'].append(results_pose2d_converted)

        # extract and pad input pose2d sequence
        pose_results_2d = self._pose_seq_extractor(
            self._buffer['pose_est_results_list'],
            frame_idx=index if self._video_input else 0)
        causal = self.cfg.test_dataloader.dataset.get('causal', False)
        target_idx = -1 if causal else len(pose_results_2d) // 2

        stats_info = self.model.dataset_meta.get('stats_info', {})
        bbox_center = stats_info.get('bbox_center', None)
        bbox_scale = stats_info.get('bbox_scale', None)

        for i, pose_res in enumerate(pose_results_2d):
            for j, data_sample in enumerate(pose_res):
                kpts = data_sample.pred_instances.keypoints
                bboxes = data_sample.pred_instances.bboxes
                keypoints = []
                for k in range(len(kpts)):
                    kpt = kpts[k]
                    if norm_pose_2d:
                        bbox = bboxes[k]
                        center = np.array([[(bbox[0] + bbox[2]) / 2,
                                            (bbox[1] + bbox[3]) / 2]])
                        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                        keypoints.append((kpt[:, :2] - center) / scale *
                                         bbox_scale + bbox_center)
                    else:
                        keypoints.append(kpt[:, :2])
                pose_results_2d[i][j].pred_instances.keypoints = np.array(
                    keypoints)
        pose_sequences_2d = collate_pose_sequence(pose_results_2d, True,
                                                  target_idx)
        if not pose_sequences_2d:
            return []

        data_list = []
        for i, pose_seq in enumerate(pose_sequences_2d):
            data_info = dict()

            keypoints_2d = pose_seq.pred_instances.keypoints
            keypoints_2d = np.squeeze(
                keypoints_2d,
                axis=0) if keypoints_2d.ndim == 4 else keypoints_2d

            T, K, C = keypoints_2d.shape

            data_info['keypoints'] = keypoints_2d
            data_info['keypoints_visible'] = np.ones((
                T,
                K,
            ),
                                                     dtype=np.float32)
            data_info['lifting_target'] = np.zeros((K, 3), dtype=np.float32)
            data_info['lifting_target_visible'] = np.ones((K, 1),
                                                          dtype=np.float32)
            data_info['camera_param'] = dict(w=width, h=height)

            data_info.update(self.model.dataset_meta)
            data_info = self.pipeline(data_info)
            data_info['data_samples'].set_field(
                img_path, 'img_path', field_type='metainfo')
            data_list.append(data_info)

        return data_list

    @torch.no_grad()
    def forward(self,
                inputs: Union[dict, tuple],
                rebase_keypoint_height: bool = False):
        """Perform forward pass through the model and process the results.

        Args:
            inputs (Union[dict, tuple]): The inputs for the model.
            rebase_keypoint_height (bool, optional): Flag to rebase the
                height of the keypoints (z-axis). Defaults to False.

        Returns:
            list: A list of data samples, each containing the model's output
                results.
        """

        pose_lift_results = self.model.test_step(inputs)

        # Post-processing of pose estimation results
        pose_est_results_converted = self._buffer['pose_est_results_list'][-1]
        for idx, pose_lift_res in enumerate(pose_lift_results):
            # Update track_id from the pose estimation results
            pose_lift_res.track_id = pose_est_results_converted[idx].get(
                'track_id', 1e4)

            # Invert x and z values of the keypoints
            keypoints = pose_lift_res.pred_instances.keypoints
            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            # If rebase_keypoint_height is True, adjust z-axis values
            if rebase_keypoint_height:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[idx].pred_instances.keypoints = keypoints

        pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        data_samples = [merge_data_samples(pose_lift_results)]
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

        self.update_model_visualizer_settings(**kwargs)

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

        inputs = self.preprocess(
            inputs, batch_size=batch_size, **preprocess_kwargs)

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
        self._buffer.clear()

    def visualize(self,
                  inputs: list,
                  preds: List[PoseDataSample],
                  return_vis: bool = False,
                  show: bool = False,
                  draw_bbox: bool = False,
                  wait_time: float = 0,
                  radius: int = 3,
                  thickness: int = 1,
                  kpt_thr: float = 0.3,
                  vis_out_dir: str = '',
                  window_name: str = '',
                  window_close_event_handler: Optional[Callable] = None
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
        det_kpt_color = self.pose2d_model.visualizer.kpt_color
        det_dataset_skeleton = self.pose2d_model.visualizer.skeleton
        det_dataset_link_color = self.pose2d_model.visualizer.link_color
        self.visualizer.det_kpt_color = det_kpt_color
        self.visualizer.det_dataset_skeleton = det_dataset_skeleton
        self.visualizer.det_dataset_link_color = det_dataset_link_color

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img = mmcv.imread(single_input, channel_order='rgb')
            elif isinstance(single_input, np.ndarray):
                img = mmcv.bgr2rgb(single_input)
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            # since visualization and inference utilize the same process,
            # the wait time is reduced when a video input is utilized,
            # thereby eliminating the issue of inference getting stuck.
            wait_time = 1e-5 if self._video_input else wait_time

            visualization = self.visualizer.add_datasample(
                window_name,
                img,
                data_sample=pred,
                det_data_sample=self._buffer['pose2d_results'],
                draw_gt=False,
                draw_bbox=draw_bbox,
                show=show,
                wait_time=wait_time,
                kpt_thr=kpt_thr)
            results.append(visualization)

            if vis_out_dir:
                out_img = mmcv.rgb2bgr(visualization)
                _, file_extension = os.path.splitext(vis_out_dir)
                if file_extension:
                    dir_name = os.path.dirname(vis_out_dir)
                    file_name = os.path.basename(vis_out_dir)
                else:
                    dir_name = vis_out_dir
                    file_name = None
                mkdir_or_exist(dir_name)

                if self._video_input:

                    if self.video_info['writer'] is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        if file_name is None:
                            file_name = os.path.basename(
                                self.video_info['name'])
                        out_file = join_path(dir_name, file_name)
                        self.video_info['writer'] = cv2.VideoWriter(
                            out_file, fourcc, self.video_info['fps'],
                            (visualization.shape[1], visualization.shape[0]))
                    self.video_info['writer'].write(out_img)

                else:
                    img_name = os.path.basename(pred.metainfo['img_path'])
                    file_name = file_name if file_name else img_name
                    out_file = join_path(dir_name, file_name)
                    mmcv.imwrite(out_img, out_file)

        if return_vis:
            return results
        else:
            return []
