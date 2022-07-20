# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random

import numpy as np
import torch
from mmengine import build_from_cfg, is_seq_of
from torch.utils.data.dataset import ConcatDataset

from mmpose.registry import DATASETS

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def _concat_dataset(cfg, default_args=None):
    types = cfg['type']
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    dataset_infos = cfg.get('dataset_info', None)

    num_joints = cfg['data_cfg'].get('num_joints', None)
    dataset_channel = cfg['data_cfg'].get('dataset_channel', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['ann_file'] = ann_files[i]

        if isinstance(types, (list, tuple)):
            cfg_copy['type'] = types[i]
        if isinstance(img_prefixes, (list, tuple)):
            cfg_copy['img_prefix'] = img_prefixes[i]
        if isinstance(dataset_infos, (list, tuple)):
            cfg_copy['dataset_info'] = dataset_infos[i]

        if isinstance(num_joints, (list, tuple)):
            cfg_copy['data_cfg']['num_joints'] = num_joints[i]

        if is_seq_of(dataset_channel, list):
            cfg_copy['data_cfg']['dataset_channel'] = dataset_channel[i]

        datasets.append(build_dataset(cfg_copy, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    from .dataset_wrappers import RepeatDataset

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
