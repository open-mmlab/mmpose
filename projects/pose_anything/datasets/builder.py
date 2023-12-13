from mmengine.dataset import RepeatDataset
from mmengine.registry import build_from_cfg
from torch.utils.data.dataset import ConcatDataset

from mmpose.datasets.builder import DATASETS


def _concat_cfg(cfg):
    replace = ['ann_file', 'img_prefix']
    channels = ['num_joints', 'dataset_channel']
    concat_cfg = []
    for i in range(len(cfg['type'])):
        cfg_tmp = cfg.deepcopy()
        cfg_tmp['type'] = cfg['type'][i]
        for item in replace:
            assert item in cfg_tmp
            assert len(cfg['type']) == len(cfg[item]), (cfg[item])
            cfg_tmp[item] = cfg[item][i]
        for item in channels:
            assert item in cfg_tmp['data_cfg']
            assert len(cfg['type']) == len(cfg['data_cfg'][item])
            cfg_tmp['data_cfg'][item] = cfg['data_cfg'][item][i]
        concat_cfg.append(cfg_tmp)
    return concat_cfg


def _check_vaild(cfg):
    replace = ['num_joints', 'dataset_channel']
    if isinstance(cfg['data_cfg'][replace[0]], (list, tuple)):
        for item in replace:
            cfg['data_cfg'][item] = cfg['data_cfg'][item][0]
    return cfg


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    if isinstance(cfg['type'],
                  (list, tuple)):  # In training, type=TransformerPoseDataset
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in _concat_cfg(cfg)])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    else:
        cfg = _check_vaild(cfg)
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset
