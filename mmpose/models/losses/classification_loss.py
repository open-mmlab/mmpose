# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss."""

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.binary_cross_entropy
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target, reduction='none')
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = (loss * target_weight).mean()
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.

    Modified from `the official implementation
    <https://github.com/leeyegy/SimCC>`_.
    """

    def __init__(self, use_target_weight=True):
        super(KLDiscretLoss, self).__init__()

        self.use_target_weight = use_target_weight
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): _description_
            gt_simcc (Tuple[Tensor, Tensor]): _description_
            target_weight (Tensor): _description_
        """
        output_x, output_y = pred_simcc
        target_x, target_y = gt_simcc
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_x_gt = target_x[:, idx].squeeze()
            coord_y_gt = target_y[:, idx].squeeze()

            if self.use_target_weight:
                weight = target_weight[:, idx].squeeze()
            else:
                weight = 1.
            
            loss += (
                self.criterion(coord_x_pred, coord_x_gt).mul(weight).sum())
            loss += (
                self.criterion(coord_y_pred, coord_y_gt).mul(weight).sum())

        return loss / num_joints


@MODELS.register_module()
class KLLossWithLabelSmoothing(nn.Module):
    """Discrete KL Divergence Loss for SimCC with one-hot or Standard Label
    Smoothing.

    Modified from `the official implementation
    <https://github.com/leeyegy/SimCC>`_.
    """

    def __init__(self, label_smoothing=0.0, use_target_weight=True):
        super(KLLossWithLabelSmoothing, self).__init__()

        self.use_target_weight = use_target_weight
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]

        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)
        device = dec_outs.device

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()

            # Do label smoothing, shape is [M]
            one_hot = self._smooth_label(num_tokens).to(device)

            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]

            # after tdata.unsqueeze(1) , tdata shape is [N,1]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            gtruth = tmp_.detach()

        loss = torch.sum(self.criterion_(scores, gtruth), dim=1)

        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        output_x, output_y = pred_simcc  # B, K, Wx
        target_x, target_y = gt_simcc  # B, K
        target = torch.cat([target_x[..., None], target_y[..., None]], dim=-1)
        batch_size = output_x.size(0)
        num_joints = output_x.size(1)

        loss = 0.

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx]
            coord_y_pred = output_y[:, idx]
            coord_gt = target[:, idx]
            
            if self.use_target_weight:
                weight = target_weight[:, idx]
            else:
                weight = 1.

            loss += self.criterion(coord_x_pred,
                                   coord_gt[:, 0]).mul(weight).sum()
            loss += self.criterion(coord_y_pred,
                                   coord_gt[:, 1]).mul(weight).sum()

        return loss / batch_size
