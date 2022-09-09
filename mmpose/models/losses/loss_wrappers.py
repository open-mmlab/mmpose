# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class MultipleLossWrapper(nn.Module):
    """A wrapper to combine multiple losses together with different factors.

    Args:
        losses (list): List of Loss Config
        factors (list): List of Loss factors
    """

    def __init__(self, losses: list, factors: list):
        super().__init__()

        assert len(losses) == len(factors), (
            'len(losses) must be equal to len(factors')

        self.factors = factors
        self.num_losses = len(factors)

        loss_modules = []
        for loss_cfg in losses:
            t_loss = MODELS.build(loss_cfg)
            loss_modules.append(t_loss)
        self.loss_modules = nn.ModuleList(loss_modules)

    def forward(self,
                input_list,
                target_list,
                keypoint_weights=None,
                epoch_facotrs=None):
        assert isinstance(input_list, list), ''
        assert isinstance(target_list, list), ''
        assert len(input_list) == len(target_list), ''

        loss_sum = 0.
        for i in range(self.num_losses):
            input_i = input_list[i]
            target_i = target_list[i]

            loss_i = self.loss_modules[i](input_i, target_i, keypoint_weights)

            if epoch_facotrs:
                loss_sum += loss_i * epoch_facotrs[i]
            else:
                loss_sum += loss_i * self.factors[i]

        return loss_sum
