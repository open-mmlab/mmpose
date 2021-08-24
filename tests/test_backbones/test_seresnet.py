# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import AvgPool2d
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.models.backbones import SEResNet
from mmpose.models.backbones.resnet import ResLayer
from mmpose.models.backbones.seresnet import SEBottleneck, SELayer


def all_zeros(modules):
    """Check if the weight(and bias) is all zero."""
    weight_zero = torch.equal(modules.weight.data,
                              torch.zeros_like(modules.weight.data))
    if hasattr(modules, 'bias'):
        bias_zero = torch.equal(modules.bias.data,
                                torch.zeros_like(modules.bias.data))
    else:
        bias_zero = True

    return weight_zero and bias_zero


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_selayer():
    # Test selayer forward
    layer = SELayer(64)
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test selayer forward with different ratio
    layer = SELayer(64, ratio=8)
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_bottleneck():

    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        SEBottleneck(64, 64, style='tensorflow')

    # Test SEBottleneck with checkpoint forward
    block = SEBottleneck(64, 64, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test Bottleneck style
    block = SEBottleneck(64, 256, stride=2, style='pytorch')
    assert block.conv1.stride == (1, 1)
    assert block.conv2.stride == (2, 2)
    block = SEBottleneck(64, 256, stride=2, style='caffe')
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)

    # Test Bottleneck forward
    block = SEBottleneck(64, 64)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_res_layer():
    # Test ResLayer of 3 Bottleneck w\o downsample
    layer = ResLayer(SEBottleneck, 3, 64, 64, se_ratio=16)
    assert len(layer) == 3
    assert layer[0].conv1.in_channels == 64
    assert layer[0].conv1.out_channels == 16
    for i in range(1, len(layer)):
        assert layer[i].conv1.in_channels == 64
        assert layer[i].conv1.out_channels == 16
    for i in range(len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test ResLayer of 3 SEBottleneck with downsample
    layer = ResLayer(SEBottleneck, 3, 64, 256, se_ratio=16)
    assert layer[0].downsample[0].out_channels == 256
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 56, 56])

    # Test ResLayer of 3 SEBottleneck with stride=2
    layer = ResLayer(SEBottleneck, 3, 64, 256, stride=2, se_ratio=8)
    assert layer[0].downsample[0].out_channels == 256
    assert layer[0].downsample[0].stride == (2, 2)
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 28, 28])

    # Test ResLayer of 3 SEBottleneck with stride=2 and average downsample
    layer = ResLayer(
        SEBottleneck, 3, 64, 256, stride=2, avg_down=True, se_ratio=8)
    assert isinstance(layer[0].downsample[0], AvgPool2d)
    assert layer[0].downsample[1].out_channels == 256
    assert layer[0].downsample[1].stride == (1, 1)
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 28, 28])


def test_seresnet():
    """Test resnet backbone."""
    with pytest.raises(KeyError):
        # SEResNet depth should be in [50, 101, 152]
        SEResNet(20)

    with pytest.raises(AssertionError):
        # In SEResNet: 1 <= num_stages <= 4
        SEResNet(50, num_stages=0)

    with pytest.raises(AssertionError):
        # In SEResNet: 1 <= num_stages <= 4
        SEResNet(50, num_stages=5)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations) == num_stages
        SEResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = SEResNet(50)
        model.init_weights(pretrained=0)

    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        SEResNet(50, style='tensorflow')

    # Test SEResNet50 norm_eval=True
    model = SEResNet(50, norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test SEResNet50 with torchvision pretrained weight
    model = SEResNet(depth=50, norm_eval=True)
    model.init_weights('torchvision://resnet50')
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test SEResNet50 with first stage frozen
    frozen_stages = 1
    model = SEResNet(50, frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    assert model.norm1.training is False
    for layer in [model.conv1, model.norm1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test SEResNet50 with BatchNorm forward
    model = SEResNet(50, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    # Test SEResNet50 with layers 1, 2, 3 out forward
    model = SEResNet(50, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])

    # Test SEResNet50 with layers 3 (top feature maps) out forward
    model = SEResNet(50, out_indices=(3, ))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size([1, 2048, 7, 7])

    # Test SEResNet50 with checkpoint forward
    model = SEResNet(50, out_indices=(0, 1, 2, 3), with_cp=True)
    for m in model.modules():
        if isinstance(m, SEBottleneck):
            assert m.with_cp
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    # Test SEResNet50 zero initialization of residual
    model = SEResNet(50, out_indices=(0, 1, 2, 3), zero_init_residual=True)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, SEBottleneck):
            assert all_zeros(m.norm3)
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])
