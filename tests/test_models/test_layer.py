# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_upsample_layer


def test_build_upsample_layer():
    layer1 = nn.ConvTranspose2d(
        in_channels=3,
        out_channels=10,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        bias=False)

    layer2 = build_upsample_layer(
        dict(type='deconv'),
        in_channels=3,
        out_channels=10,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        bias=False)
    layer2.load_state_dict(layer1.state_dict())

    input_shape = (1, 3, 32, 32)
    inputs = _demo_inputs(input_shape)
    out1 = layer1(inputs)
    out2 = layer2(inputs)
    assert torch.equal(out1, out2)


def test_build_conv_layer():
    layer1 = nn.Conv2d(
        in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)

    layer2 = build_conv_layer(
        cfg=dict(type='Conv2d'),
        in_channels=3,
        out_channels=10,
        kernel_size=3,
        stride=1,
        padding=1)

    layer2.load_state_dict(layer1.state_dict())

    input_shape = (1, 3, 32, 32)
    inputs = _demo_inputs(input_shape)
    out1 = layer1(inputs)
    out2 = layer2(inputs)
    assert torch.equal(out1, out2)


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    Returns:
        Random input tensor with the size of input_shape.
    """
    inps = np.random.random(input_shape)
    inps = torch.FloatTensor(inps)
    return inps
