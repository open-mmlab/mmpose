import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss."""

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.binary_cross_entropy
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            loss = self.criterion(output, target, reduction='none')
            loss = (loss * target_weight).mean()
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight
