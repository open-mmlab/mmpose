# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from mmpose.structures.bbox import bbox_overlaps


@MODELS.register_module()
class IoULoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        eps (float): Epsilon to avoid log(0).
        loss_weight (float): Weight of the loss. Default: 1.0.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 reduction='mean',
                 mode='log',
                 eps: float = 1e-16,
                 loss_weight=1.):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'the argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        assert mode in ('linear', 'square', 'log'), f'the argument ' \
            f'`reduction` should be either \'linear\', \'square\' or ' \
            f'\'log\', but got {mode}'

        self.reduction = reduction
        self.criterion = partial(F.cross_entropy, reduction='none')
        self.loss_weight = loss_weight
        self.mode = mode
        self.eps = eps

    def forward(self, output, target):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
        """
        ious = bbox_overlaps(
            output, target, is_aligned=True).clamp(min=self.eps)

        if self.mode == 'linear':
            loss = 1 - ious
        elif self.mode == 'square':
            loss = 1 - ious.pow(2)
        elif self.mode == 'log':
            loss = -ious.log()
        else:
            raise NotImplementedError

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight
