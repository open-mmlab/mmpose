import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """SmoothL1Loss loss ."""

    def __init__(self):
        super().__init__()
        self.criterion = F.smooth_l1_loss

    def forward(self, output, target, target_weight):
        """Forward function."""
        num_joints = output.size(1)
        loss = self.criterion(
            output.mul(target_weight), target.mul(target_weight))

        return loss / num_joints
