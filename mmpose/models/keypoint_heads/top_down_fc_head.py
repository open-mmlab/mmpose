import torch
import torch.nn as nn

from mmpose.models.registry import HEADS


@HEADS.register_module()
class TopDownFcHead(nn.Module):
    """Top-down pose head with fully connected layers.

    paper ref: Alexander Toshev and Christian Szegedy,
    ``DeepPose: Human Pose Estimation via Deep Neural Networks.''.

    Args:
        in_channels (int): Number of input channels
        num_hidden (List(int)): Number of hidden nodes.
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels, num_hidden=(4096, 4096)):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.reg_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.regression = nn.Linear(self.in_channels, self.out_channels * 2)

    def forward(self, x):
        """Forward function."""
        x_reg = self.reg_layer(x)
        x_reg = self.reg_avgpool(x_reg)
        x_reg = torch.flatten(x_reg, 1)
        regression = self.regression(x_reg)
        regression = regression.reshape(regression.shape[0], -1, 2)

        return regression

    def init_weights(self):
        """Initialize model weights."""
        pass
