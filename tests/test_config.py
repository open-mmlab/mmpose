# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, exists, join, relpath

import torch
from mmcv.runner import build_optimizer


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmpose
        repo_dpath = dirname(dirname(mmpose.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def test_config_build_detector():
    """Test that all detection models defined in the configs can be
    initialized."""
    from mmcv import Config

    from mmpose.models import build_posenet

    config_dpath = _get_config_directory()
    print(f'Found config_dpath = {config_dpath}')

    import glob
    config_fpaths = list(glob.glob(join(config_dpath, '**', '*.py')))
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]

    print(f'Using {len(config_names)} config files')

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)

        print(f'Building detector, config_fpath = {config_fpath}')

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model:
            config_mod.model['pretrained'] = None

        detector = build_posenet(config_mod.model)
        assert detector is not None

        optimizer = build_optimizer(detector, config_mod.optimizer)
        assert isinstance(optimizer, torch.optim.Optimizer)
