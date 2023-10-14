# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from PIL import Image

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.builder import build_pose_estimator
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy


def dataset_meta_from_config(config: Config,
                             dataset_mode: str = 'train') -> Optional[dict]:
    """Get dataset metainfo from the model config.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        dataset_mode (str): Specify the dataset of which to get the metainfo.
            Options are ``'train'``, ``'val'`` and ``'test'``. Defaults to
            ``'train'``

    Returns:
        dict, optional: The dataset metainfo. See
        ``mmpose.datasets.datasets.utils.parse_pose_metainfo`` for details.
        Return ``None`` if failing to get dataset metainfo from the config.
    """
    try:
        if dataset_mode == 'train':
            dataset_cfg = config.train_dataloader.dataset
        elif dataset_mode == 'val':
            dataset_cfg = config.val_dataloader.dataset
        elif dataset_mode == 'test':
            dataset_cfg = config.test_dataloader.dataset
        else:
            raise ValueError(
                f'Invalid dataset {dataset_mode} to get metainfo. '
                'Should be one of "train", "val", or "test".')

        if 'metainfo' in dataset_cfg:
            metainfo = dataset_cfg.metainfo
        else:
            import mmpose.datasets.datasets  # noqa: F401, F403
            from mmpose.registry import DATASETS

            dataset_class = dataset_cfg.type if isinstance(
                dataset_cfg.type, type) else DATASETS.get(dataset_cfg.type)
            metainfo = dataset_class.METAINFO

        metainfo = parse_pose_metainfo(metainfo)

    except AttributeError:
        metainfo = None

    return metainfo


def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None) -> nn.Module:
    """Initialize a pose estimator from a config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Defaults to ``None``
        device (str): The device where the anchors will be put on.
            Defaults to ``'cuda:0'``.
        cfg_options (dict, optional): Options to override some settings in
            the used config. Defaults to ``None``

    Returns:
        nn.Module: The constructed pose estimator.
    """

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None

    # register all modules in mmpose into the registries
    scope = config.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    model = build_pose_estimator(config.model)
    model = revert_sync_batchnorm(model)
    # get dataset_meta in this priority: checkpoint > config > default (COCO)
    dataset_meta = None

    if checkpoint is not None:
        ckpt = load_checkpoint(model, checkpoint, map_location='cpu')

        if 'dataset_meta' in ckpt.get('meta', {}):
            # checkpoint from mmpose 1.x
            dataset_meta = ckpt['meta']['dataset_meta']

    if dataset_meta is None:
        dataset_meta = dataset_meta_from_config(config, dataset_mode='train')

    if dataset_meta is None:
        warnings.simplefilter('once')
        warnings.warn('Can not load dataset_meta from the checkpoint or the '
                      'model config. Use COCO metainfo by default.')
        dataset_meta = parse_pose_metainfo(
            dict(from_file='configs/_base_/datasets/coco.py'))

    model.dataset_meta = dataset_meta

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_topdown(model: nn.Module,
                      img: Union[np.ndarray, str],
                      bboxes: Optional[Union[List, np.ndarray]] = None,
                      bbox_format: str = 'xyxy') -> List[PoseDataSample]:
    """Inference image with a top-down pose estimator.

    Args:
        model (nn.Module): The top-down pose estimator
        img (np.ndarray | str): The loaded image or image file to inference
        bboxes (np.ndarray, optional): The bboxes in shape (N, 4), each row
            represents a bbox. If not given, the entire image will be regarded
            as a single bbox area. Defaults to ``None``
        bbox_format (str): The bbox format indicator. Options are ``'xywh'``
            and ``'xyxy'``. Defaults to ``'xyxy'``

    Returns:
        List[:obj:`PoseDataSample`]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """
    scope = model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    if bboxes is None or len(bboxes) == 0:
        # get bbox from the image size
        if isinstance(img, str):
            w, h = Image.open(img).size
        else:
            h, w = img.shape[:2]

        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
    else:
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        assert bbox_format in {'xyxy', 'xywh'}, \
            f'Invalid bbox_format "{bbox_format}".'

        if bbox_format == 'xywh':
            bboxes = bbox_xywh2xyxy(bboxes)

    # construct batch data samples
    data_list = []
    for bbox in bboxes:
        if isinstance(img, str):
            data_info = dict(img_path=img)
        else:
            data_info = dict(img=img)
        data_info['bbox'] = bbox[None]  # shape (1, 4)
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []

    return results


def inference_bottomup(model: nn.Module, img: Union[np.ndarray, str]):
    """Inference image with a bottom-up pose estimator.

    Args:
        model (nn.Module): The bottom-up pose estimator
        img (np.ndarray | str): The loaded image or image file to inference

    Returns:
        List[:obj:`PoseDataSample`]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # prepare data batch
    if isinstance(img, str):
        data_info = dict(img_path=img)
    else:
        data_info = dict(img=img)
    data_info.update(model.dataset_meta)
    data = pipeline(data_info)
    batch = pseudo_collate([data])

    with torch.no_grad():
        results = model.test_step(batch)

    return results


def collect_multi_frames(video, frame_id, indices, online=False):
    """Collect multi frames from the video.

    Args:
        video (mmcv.VideoReader): A VideoReader of the input video file.
        frame_id (int): index of the current frame
        indices (list(int)): index offsets of the frames to collect
        online (bool): inference mode, if set to True, can not use future
            frame information.

    Returns:
        list(ndarray): multi frames collected from the input video file.
    """
    num_frames = len(video)
    frames = []
    # put the current frame at first
    frames.append(video[frame_id])
    # use multi frames for inference
    for idx in indices:
        # skip current frame
        if idx == 0:
            continue
        support_idx = frame_id + idx
        # online mode, can not use future frame information
        if online:
            support_idx = np.clip(support_idx, 0, frame_id)
        else:
            support_idx = np.clip(support_idx, 0, num_frames - 1)
        frames.append(video[support_idx])

    return frames
