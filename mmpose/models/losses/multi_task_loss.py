# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class MultiTaskLoss(nn.Module):
    """Combine multi-task losses together with different factors.

    Args:
        loss_cfg_list (list): List of Loss Config
        factors (list): List of Loss factors
    """

    def __init__(self, loss_cfg_list: tuple, factors: tuple):
        super().__init__()

        assert len(loss_cfg_list) == len(factors), (
            'len(loss_cfg_list) must equal to len(factors')

        self.factors = factors
        self.num_losses = len(factors)

        losses = []
        for loss_cfg in loss_cfg_list:
            t_loss = MODELS.build(loss_cfg)
            losses.append(t_loss)
        self.losses = nn.ModuleList(losses)

    def forward(self, input_list, target_list, keypoint_weights=None):
        assert isinstance(input_list, list), ''
        assert isinstance(target_list, list), ''
        assert len(input_list) == len(target_list), ''

        loss_sum = 0.
        for i in range(self.num_losses):
            input_i = input_list[i]
            target_i = target_list[i]

            loss_i = self.losses[i](input_i, target_i, keypoint_weights)
            loss_sum += loss_i * self.factors[i]

        return loss_sum
