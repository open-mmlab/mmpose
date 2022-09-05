# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmpose.registry import HOOKS


@HOOKS.register_module()
class ModelSetEpochHook(Hook):
    """The hook that tells model the current epoch in training."""

    def __init__(self, module_name='head'):
        self.module_name = module_name

    def before_train_epoch(self, runner: Runner):
        m = getattr(runner.model, self.module_name)
        m.set_train_epoch(runner.epoch)
