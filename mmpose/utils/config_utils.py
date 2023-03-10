# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.utils.typing import ConfigDict


def convert_mmdet_test_pipeline(cfg: ConfigDict):
    from mmdet.datasets import transforms

    if 'test_dataloader' not in cfg:
        return cfg

    pipeline = cfg.test_dataloader.dataset.pipeline
    for trans in pipeline:
        if trans['type'] in dir(transforms):
            trans['type'] = 'mmdet.' + trans['type']

    return cfg
