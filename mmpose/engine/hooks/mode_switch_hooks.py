# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmpose.registry import HOOKS
from mmpose.utils.hooks import rgetattr, rsetattr


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


@HOOKS.register_module()
class RTMOModeSwitchHook(Hook):
    """A hook to switch the mode of RTMO during training.

    This hook allows for dynamic adjustments of model attributes at specified
    training epochs. It is designed to modify configurations such as turning
    off specific augmentations or changing loss functions at different stages
    of the training process.

    Args:
        epoch_attributes (Dict[str, Dict]): A dictionary where keys are epoch
        numbers and values are attribute modification dictionaries. Each
        dictionary specifies the attribute to modify and its new value.

    Example:
        epoch_attributes = {
            5: [{"attr1.subattr": new_value1}, {"attr2.subattr": new_value2}],
            10: [{"attr3.subattr": new_value3}]
        }
    """

    def __init__(self, epoch_attributes: Dict[int, Dict]):
        self.epoch_attributes = epoch_attributes

    def before_train_epoch(self, runner: Runner):
        """Method called before each training epoch.

        It checks if the current epoch is in the `epoch_attributes` mapping and
        applies the corresponding attribute changes to the model.
        """
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if epoch in self.epoch_attributes:
            for key, value in self.epoch_attributes[epoch].items():
                rsetattr(model.head, key, value)
                runner.logger.info(
                    f'Change model.head.{key} to {rgetattr(model.head, key)}')
