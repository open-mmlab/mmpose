# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path
from typing import Optional, Union

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import nn

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.builder import build_pose_estimator


def dataset_meta_from_config(config: Config,
                             dataset: str = 'train') -> Optional[dict]:
    """Get dataset metainfo from the model config.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        dataset (str): Specify the dataset of which to get the metainfo.
            Options are ``'train'``, ``'val'`` and ``'test'``. Defaults to
            ``'train'``

    Returns:
        dict, optional: The dataset metainfo. See
        ``mmpose.datasets.datasets.parse_pose_metainfo`` for more details.
        Return ``None`` if fail to get dataset metainfo from the config.
    """
    try:
        if dataset == 'train':
            dataset_cfg = config.train_dataloader.dataset
        elif dataset == 'val':
            dataset_cfg = config.val_dataloader.dataset
        elif dataset == 'test':
            dataset_cfg = config.test_dataloader.dataset
        else:
            raise ValueError(
                f'Invalid dataset {dataset} to get metainfo. Should be one of '
                '"train", "val", or "test".')

        if 'metainfo' in dataset_cfg:
            metainfo = dataset_cfg.metainfo
        else:
            import mmpose.datasets.datasets  # noqa: F401, F403
            from mmpose.registry import DATASETS

            dataset_class = DATASETS.get(dataset_cfg.type)
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

    model = build_pose_estimator(config.model)

    if checkpoint is not None:
        ckpt = load_checkpoint(model, checkpoint, map_location='cpu')

        if 'dataset_meta' in ckpt.get('meta', {}):
            # checkpoint from mmpose 1.x
            model.dataset_meta = ckpt['meta']['dataset_meta']
        else:
            dataset_meta = dataset_meta_from_config(config, dataset='test')

            if dataset_meta is None:
                warnings.simplefilter('once')
                warnings.warn(
                    'Can not load dataset_meta from the checkpoint or the '
                    'model config. Use COCO metainfo by default.')
                dataset_meta = parse_pose_metainfo(
                    dict(from_file='configs/_base_/datasets/coco.py'))

            model.dataset_meta = dataset_meta

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model
