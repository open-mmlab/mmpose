# Copyright (c) OpenMMLab. All rights reserved.
import types
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmpose.utils.typing import OptConfigType


class RepVGGBlock(BaseModule):
    """A block in RepVGG architecture, supporting optional normalization in the
    identity branch.

    This block consists of 3x3 and 1x1 convolutions, with an optional identity
    shortcut branch that includes normalization.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): The stride of the block. Defaults to 1.
        padding (int): The padding of the block. Defaults to 1.
        dilation (int): The dilation of the block. Defaults to 1.
        groups (int): The groups of the block. Defaults to 1.
        padding_mode (str): The padding mode of the block. Defaults to 'zeros'.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to dict(type='BN').
        act_cfg (dict): The config dict for activation layers.
            Defaults to dict(type='ReLU').
        without_branch_norm (bool): Whether to skip branch_norm.
            Defaults to True.
        init_cfg (dict): The config dict for initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU'),
                 without_branch_norm: bool = True,
                 init_cfg: OptConfigType = None):
        super(RepVGGBlock, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # judge if input shape and output shape are the same.
        # If true, add a normalized identity shortcut.
        self.branch_norm = None
        if out_channels == in_channels and stride == 1 and \
                padding == dilation and not without_branch_norm:
            self.branch_norm = build_norm_layer(norm_cfg, in_channels)[1]

        self.branch_3x3 = ConvModule(
            self.in_channels,
            self.out_channels,
            3,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.branch_1x1 = ConvModule(
            self.in_channels,
            self.out_channels,
            1,
            groups=self.groups,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the RepVGG block.

        The output is the sum of 3x3 and 1x1 convolution outputs,
        along with the normalized identity branch output, followed by
        activation.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        if self.branch_norm is None:
            branch_norm_out = 0
        else:
            branch_norm_out = self.branch_norm(x)

        out = self.branch_3x3(x) + self.branch_1x1(x) + branch_norm_out

        out = self.act(out)

        return out

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tensor:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0

        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.branch_3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.branch_1x1)
        kernelid, biasid = (0, 0) if self.branch_norm is None else \
            self._fuse_bn_tensor(self.branch_norm)

        return (kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
                bias3x3 + bias1x1 + biasid)

    def switch_to_deploy(self, test_cfg: Optional[Dict] = None):
        """Switches the block to deployment mode.

        In deployment mode, the block uses a single convolution operation
        derived from the equivalent kernel and bias, replacing the original
        branches. This reduces computational complexity during inference.
        """
        if getattr(self, 'deploy', False):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_reparam = nn.Conv2d(
            in_channels=self.branch_3x3.conv.in_channels,
            out_channels=self.branch_3x3.conv.out_channels,
            kernel_size=self.branch_3x3.conv.kernel_size,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True)
        self.conv_reparam.weight.data = kernel
        self.conv_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('branch_3x3')
        self.__delattr__('branch_1x1')
        if hasattr(self, 'branch_norm'):
            self.__delattr__('branch_norm')

        def _forward(self, x):
            return self.act(self.conv_reparam(x))

        self.forward = types.MethodType(_forward, self)

        self.deploy = True
