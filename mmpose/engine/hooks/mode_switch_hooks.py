# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmpose.registry import HOOKS


@HOOKS.register_module()
class YOLOXPoseModeSwitchHook(Hook):
    """Switch the mode of YOLOX-Pose during training.

    This hook:
    1) Turns off mosaic and mixup data augmentation.
    2) Uses instance mask to assist positive anchor selection.
    3) Uses auxiliary L1 loss in the head.

    Args:
        num_last_epochs (int): The number of last epochs at the end of
            training to close the data augmentation and switch to L1 loss.
            Defaults to 20.
        new_train_dataset (dict): New training dataset configuration that
            will be used in place of the original training dataset. Defaults
            to None.
        new_train_pipeline (Sequence[dict]): New data augmentation pipeline
            configuration that will be used in place of the original pipeline
            during training. Defaults to None.
    """

    def __init__(self,
                 num_last_epochs: int = 20,
                 new_train_dataset: dict = None,
                 new_train_pipeline: Sequence[dict] = None):
        self.num_last_epochs = num_last_epochs
        self.new_train_dataset = new_train_dataset
        self.new_train_pipeline = new_train_pipeline

    def _modify_dataloader(self, runner: Runner):
        """Modify dataloader with new dataset and pipeline configurations."""
        runner.logger.info(f'New Pipeline: {self.new_train_pipeline}')

        train_dataloader_cfg = copy.deepcopy(runner.cfg.train_dataloader)
        if self.new_train_dataset:
            train_dataloader_cfg.dataset = self.new_train_dataset
        if self.new_train_pipeline:
            train_dataloader_cfg.dataset.pipeline = self.new_train_pipeline

        new_train_dataloader = Runner.build_dataloader(train_dataloader_cfg)
        runner.train_loop.dataloader = new_train_dataloader
        runner.logger.info('Recreated the dataloader!')

    def before_train_epoch(self, runner: Runner):
        """Close mosaic and mixup augmentation, switch to use L1 loss."""
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if epoch + 1 == runner.max_epochs - self.num_last_epochs:
            self._modify_dataloader(runner)
            runner.logger.info('Added additional reg loss now!')
            model.head.use_aux_loss = True
