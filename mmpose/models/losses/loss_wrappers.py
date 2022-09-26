# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class MultipleLossWrapper(nn.Module):
    """A wrapper to collect multiple loss functions together and return a list
    of losses in the same order.

    Args:
        losses (list): List of Loss Config
    """

    def __init__(self, losses: list):
        super().__init__()
        self.num_losses = len(losses)

        loss_modules = []
        for loss_cfg in losses:
            t_loss = MODELS.build(loss_cfg)
            loss_modules.append(t_loss)
        self.loss_modules = nn.ModuleList(loss_modules)

    def forward(self, input_list, target_list, keypoint_weights=None):
        assert len(input_list) == len(target_list), ''

        losses = []
        for i in range(self.num_losses):
            input_i = input_list[i]
            target_i = target_list[i]

            loss_i = self.loss_modules[i](input_i, target_i, keypoint_weights)
            losses.append(loss_i)

        return losses
