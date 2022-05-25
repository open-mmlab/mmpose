# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ModelSetEpochHook(Hook):
    """Set `epoch` attribute for model while training."""

    def __init__(self):
        pass

    def before_epoch(self, runner):
        runner.model.module.set_train_epoch(runner.epoch + 1)
