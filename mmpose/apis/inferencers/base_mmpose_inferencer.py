# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from typing import (Any, Callable, Dict, Generator, List, Optional, Sequence,
                    Union)

import cv2
import mmcv
import mmengine
import numpy as np
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import BaseInferencer
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmengine.structures import InstanceData

from mmpose.apis.inference import dataset_meta_from_config
from mmpose.structures import PoseDataSample, split_instances

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class BaseMMPoseInferencer(BaseInferencer):
    """The base class for MMPose inferencers."""

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
        self._video_input = False

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
                inputs.sort()
            else:
                # if inputs is a path to a video file, it will be converted
                # to a list containing separated frame filenames
                input_type = mimetypes.guess_type(inputs)[0].split('/')[0]
                if input_type == 'video':
                    self._video_input = True
                    # split video frames into a temporary folder
                    frame_folder = tempfile.TemporaryDirectory()
                    video = mmcv.VideoReader(inputs)
                    self.video_info = dict(
                        fps=video.fps,
                        name=os.path.basename(inputs),
                        frame_folder=frame_folder)
                    video.cvt2frames(frame_folder.name, show_progress=False)
                    frames = sorted(list_dir_or_file(frame_folder.name))
                    inputs = [join_path(frame_folder.name, f) for f in frames]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _get_webcam_inputs(self, inputs: str) -> Generator:
        """Sets up and returns a generator function that reads frames from a
        webcam input. The generator function returns a new frame each time it
        is iterated over.

        Args:
            inputs (str): A string describing the webcam input, in the format
                "webcam:id".

        Returns:
            A generator function that yields frames from the webcam input.

        Raises:
            ValueError: If the inputs string is not in the expected format.
        """

        # Ensure the inputs string is in the expected format.
        inputs = inputs.lower()
        assert inputs.startswith('webcam'), f'Expected input to start with ' \
            f'"webcam", but got "{inputs}"'

        # Parse the camera ID from the inputs string.
        inputs_ = inputs.split(':')
        if len(inputs_) == 1:
            camera_id = 0
        elif len(inputs_) == 2 and str.isdigit(inputs_[1]):
            camera_id = int(inputs_[1])
        else:
            raise ValueError(
                f'Expected webcam input to have format "webcam:id", '
                f'but got "{inputs}"')

        # Attempt to open the video capture object.
        vcap = cv2.VideoCapture(camera_id)
        if not vcap.isOpened():
            warnings.warn(f'Cannot open camera (ID={camera_id})')
            return []

        # Set video input flag and metadata.
        self._video_input = True
        self.video_info = dict(fps=10, name='webcam.mp4', frame_folder=None)

        # Set up webcam reader generator function.
        self._window_closing = False

        def _webcam_reader() -> Generator:
            while True:
                if self._window_closing:
                    vcap.release()
                    break

                ret_val, frame = vcap.read()
                if not ret_val:
                    break

                yield frame

        return _webcam_reader()

    def _visualization_window_on_close(self, event):
        self._window_closing = True

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
            data_infos = self.preprocess_single(input, index=i, **kwargs)
            # only supports inference with batch size 1
            yield self.collate_fn(data_infos), [input]

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

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img = mmcv.imread(single_input, channel_order='rgb')
            elif isinstance(single_input, np.ndarray):
                img = mmcv.bgr2rgb(single_input.copy())
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            img_name = os.path.basename(pred.metainfo['img_path'])

            if vis_out_dir:
                if self._video_input:
                    out_file = join_path(vis_out_dir, 'vis_frames', img_name)
                else:
                    out_file = join_path(vis_out_dir, img_name)
            else:
                out_file = None

            # since visualization and inference utilize the same process,
            # the wait time is reduced when a video input is utilized,
            # thereby eliminating the issue of inference getting stuck.
            wait_time = 1e-5 if self._video_input else wait_time

            window_name = window_name if window_name else img_name

            visualization = self.visualizer.add_datasample(
                window_name,
                img,
                pred,
                draw_gt=False,
                draw_bbox=draw_bbox,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                kpt_thr=kpt_thr)
            results.append(visualization)

            if show and not hasattr(self, '_window_close_cid'):
                if window_close_event_handler is None:
                    window_close_event_handler = \
                        self._visualization_window_on_close
                self._window_close_cid = \
                    self.visualizer.manager.canvas.mpl_connect(
                        'close_event',
                        window_close_event_handler
                    )

        if return_vis:
            return results
        else:
            return []

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

        if pred_out_dir != '':
            if self._video_input:
                pred_out_dir = join_path(pred_out_dir, 'pred_frames')

            for pred, data_sample in zip(result_dict['predictions'], preds):
                fname = os.path.splitext(
                    os.path.basename(
                        data_sample.metainfo['img_path']))[0] + '.json'
                mmengine.dump(
                    pred, join_path(pred_out_dir, fname), indent='  ')

        return result_dict

    def _merge_outputs(self, vis_out_dir: str, pred_out_dir: str,
                       **kwargs: Dict[str, Any]) -> None:
        """Merge the visualized frames and predicted instance outputs and save
        them.

        Args:
            vis_out_dir (str): Path to the directory where the visualized
                frames are saved.
            pred_out_dir (str): Path to the directory where the predicted
                instance outputs are saved.
            **kwargs: Other arguments that are not used in this method.
        """
        assert self._video_input

        if vis_out_dir != '':
            vis_frame_out_dir = join_path(vis_out_dir, 'vis_frames')
            if not isdir(vis_frame_out_dir) or len(
                    os.listdir(vis_frame_out_dir)) == 0:
                warnings.warn(
                    f'{vis_frame_out_dir} does not exist or is empty.')
            else:
                mmcv.frames2video(
                    vis_frame_out_dir,
                    join_path(vis_out_dir, self.video_info['name']),
                    fps=self.video_info['fps'],
                    fourcc='mp4v',
                    show_progress=False)
                shutil.rmtree(vis_frame_out_dir)

        if pred_out_dir != '':
            pred_frame_out_dir = join_path(pred_out_dir, 'pred_frames')
            if not isdir(pred_frame_out_dir) or len(
                    os.listdir(pred_frame_out_dir)) == 0:
                warnings.warn(
                    f'{pred_frame_out_dir} does not exist or is empty.')
            else:
                predictions = []
                pred_files = list_dir_or_file(pred_frame_out_dir)
                for frame_id, pred_file in enumerate(sorted(pred_files)):
                    predictions.append({
                        'frame_id':
                        frame_id,
                        'instances':
                        mmengine.load(
                            join_path(pred_frame_out_dir, pred_file))
                    })
                fname = os.path.splitext(
                    os.path.basename(self.video_info['name']))[0] + '.json'
                mmengine.dump(
                    predictions, join_path(pred_out_dir, fname), indent='  ')
                shutil.rmtree(pred_frame_out_dir)
