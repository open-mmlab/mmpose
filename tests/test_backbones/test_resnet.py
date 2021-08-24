# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmpose.models.backbones import ResNet, ResNetV1d
from mmpose.models.backbones.resnet import (BasicBlock, Bottleneck, ResLayer,
                                            get_expansion)


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (BasicBlock, Bottleneck)):
        return True
    return False


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


def test_get_expansion():
    assert get_expansion(Bottleneck, 2) == 2
    assert get_expansion(BasicBlock) == 1
    assert get_expansion(Bottleneck) == 4

    class MyResBlock(nn.Module):

        expansion = 8

    assert get_expansion(MyResBlock) == 8

    # expansion must be an integer or None
    with pytest.raises(TypeError):
        get_expansion(Bottleneck, '0')

    # expansion is not specified and cannot be inferred
    with pytest.raises(TypeError):

        class SomeModule(nn.Module):
            pass

        get_expansion(SomeModule)


def test_basic_block():
    # expansion must be 1
    with pytest.raises(AssertionError):
        BasicBlock(64, 64, expansion=2)

    # BasicBlock with stride 1, out_channels == in_channels
    block = BasicBlock(64, 64)
    assert block.in_channels == 64
    assert block.mid_channels == 64
    assert block.out_channels == 64
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 64
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv1.stride == (1, 1)
    assert block.conv2.in_channels == 64
    assert block.conv2.out_channels == 64
    assert block.conv2.kernel_size == (3, 3)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # BasicBlock with stride 1 and downsample
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128))
    block = BasicBlock(64, 128, downsample=downsample)
    assert block.in_channels == 64
    assert block.mid_channels == 128
    assert block.out_channels == 128
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 128
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv1.stride == (1, 1)
    assert block.conv2.in_channels == 128
    assert block.conv2.out_channels == 128
    assert block.conv2.kernel_size == (3, 3)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 128, 56, 56])

    # BasicBlock with stride 2 and downsample
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(128))
    block = BasicBlock(64, 128, stride=2, downsample=downsample)
    assert block.in_channels == 64
    assert block.mid_channels == 128
    assert block.out_channels == 128
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 128
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv1.stride == (2, 2)
    assert block.conv2.in_channels == 128
    assert block.conv2.out_channels == 128
    assert block.conv2.kernel_size == (3, 3)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 128, 28, 28])

    # forward with checkpointing
    block = BasicBlock(64, 64, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56, requires_grad=True)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_bottleneck():
    # style must be in ['pytorch', 'caffe']
    with pytest.raises(AssertionError):
        Bottleneck(64, 64, style='tensorflow')

    # expansion must be divisible by out_channels
    with pytest.raises(AssertionError):
        Bottleneck(64, 64, expansion=3)

    # Test Bottleneck style
    block = Bottleneck(64, 64, stride=2, style='pytorch')
    assert block.conv1.stride == (1, 1)
    assert block.conv2.stride == (2, 2)
    block = Bottleneck(64, 64, stride=2, style='caffe')
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)

    # Bottleneck with stride 1
    block = Bottleneck(64, 64, style='pytorch')
    assert block.in_channels == 64
    assert block.mid_channels == 16
    assert block.out_channels == 64
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 16
    assert block.conv1.kernel_size == (1, 1)
    assert block.conv2.in_channels == 16
    assert block.conv2.out_channels == 16
    assert block.conv2.kernel_size == (3, 3)
    assert block.conv3.in_channels == 16
    assert block.conv3.out_channels == 64
    assert block.conv3.kernel_size == (1, 1)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 56, 56)

    # Bottleneck with stride 1 and downsample
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1), nn.BatchNorm2d(128))
    block = Bottleneck(64, 128, style='pytorch', downsample=downsample)
    assert block.in_channels == 64
    assert block.mid_channels == 32
    assert block.out_channels == 128
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 32
    assert block.conv1.kernel_size == (1, 1)
    assert block.conv2.in_channels == 32
    assert block.conv2.out_channels == 32
    assert block.conv2.kernel_size == (3, 3)
    assert block.conv3.in_channels == 32
    assert block.conv3.out_channels == 128
    assert block.conv3.kernel_size == (1, 1)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 128, 56, 56)

    # Bottleneck with stride 2 and downsample
    downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128))
    block = Bottleneck(
        64, 128, stride=2, style='pytorch', downsample=downsample)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 128, 28, 28)

    # Bottleneck with expansion 2
    block = Bottleneck(64, 64, style='pytorch', expansion=2)
    assert block.in_channels == 64
    assert block.mid_channels == 32
    assert block.out_channels == 64
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 32
    assert block.conv1.kernel_size == (1, 1)
    assert block.conv2.in_channels == 32
    assert block.conv2.out_channels == 32
    assert block.conv2.kernel_size == (3, 3)
    assert block.conv3.in_channels == 32
    assert block.conv3.out_channels == 64
    assert block.conv3.kernel_size == (1, 1)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == (1, 64, 56, 56)

    # Test Bottleneck with checkpointing
    block = Bottleneck(64, 64, with_cp=True)
    block.train()
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56, requires_grad=True)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_basicblock_reslayer():
    # 3 BasicBlock w/o downsample
    layer = ResLayer(BasicBlock, 3, 32, 32)
    assert len(layer) == 3
    for i in range(3):
        assert layer[i].in_channels == 32
        assert layer[i].out_channels == 32
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 32, 56, 56)

    # 3 BasicBlock w/ stride 1 and downsample
    layer = ResLayer(BasicBlock, 3, 32, 64)
    assert len(layer) == 3
    assert layer[0].in_channels == 32
    assert layer[0].out_channels == 64
    assert layer[0].downsample is not None and len(layer[0].downsample) == 2
    assert isinstance(layer[0].downsample[0], nn.Conv2d)
    assert layer[0].downsample[0].stride == (1, 1)
    for i in range(1, 3):
        assert layer[i].in_channels == 64
        assert layer[i].out_channels == 64
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 64, 56, 56)

    # 3 BasicBlock w/ stride 2 and downsample
    layer = ResLayer(BasicBlock, 3, 32, 64, stride=2)
    assert len(layer) == 3
    assert layer[0].in_channels == 32
    assert layer[0].out_channels == 64
    assert layer[0].stride == 2
    assert layer[0].downsample is not None and len(layer[0].downsample) == 2
    assert isinstance(layer[0].downsample[0], nn.Conv2d)
    assert layer[0].downsample[0].stride == (2, 2)
    for i in range(1, 3):
        assert layer[i].in_channels == 64
        assert layer[i].out_channels == 64
        assert layer[i].stride == 1
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 64, 28, 28)

    # 3 BasicBlock w/ stride 2 and downsample with avg pool
    layer = ResLayer(BasicBlock, 3, 32, 64, stride=2, avg_down=True)
    assert len(layer) == 3
    assert layer[0].in_channels == 32
    assert layer[0].out_channels == 64
    assert layer[0].stride == 2
    assert layer[0].downsample is not None and len(layer[0].downsample) == 3
    assert isinstance(layer[0].downsample[0], nn.AvgPool2d)
    assert layer[0].downsample[0].stride == 2
    for i in range(1, 3):
        assert layer[i].in_channels == 64
        assert layer[i].out_channels == 64
        assert layer[i].stride == 1
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 64, 28, 28)


def test_bottleneck_reslayer():
    # 3 Bottleneck w/o downsample
    layer = ResLayer(Bottleneck, 3, 32, 32)
    assert len(layer) == 3
    for i in range(3):
        assert layer[i].in_channels == 32
        assert layer[i].out_channels == 32
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 32, 56, 56)

    # 3 Bottleneck w/ stride 1 and downsample
    layer = ResLayer(Bottleneck, 3, 32, 64)
    assert len(layer) == 3
    assert layer[0].in_channels == 32
    assert layer[0].out_channels == 64
    assert layer[0].stride == 1
    assert layer[0].conv1.out_channels == 16
    assert layer[0].downsample is not None and len(layer[0].downsample) == 2
    assert isinstance(layer[0].downsample[0], nn.Conv2d)
    assert layer[0].downsample[0].stride == (1, 1)
    for i in range(1, 3):
        assert layer[i].in_channels == 64
        assert layer[i].out_channels == 64
        assert layer[i].conv1.out_channels == 16
        assert layer[i].stride == 1
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 64, 56, 56)

    # 3 Bottleneck w/ stride 2 and downsample
    layer = ResLayer(Bottleneck, 3, 32, 64, stride=2)
    assert len(layer) == 3
    assert layer[0].in_channels == 32
    assert layer[0].out_channels == 64
    assert layer[0].stride == 2
    assert layer[0].conv1.out_channels == 16
    assert layer[0].downsample is not None and len(layer[0].downsample) == 2
    assert isinstance(layer[0].downsample[0], nn.Conv2d)
    assert layer[0].downsample[0].stride == (2, 2)
    for i in range(1, 3):
        assert layer[i].in_channels == 64
        assert layer[i].out_channels == 64
        assert layer[i].conv1.out_channels == 16
        assert layer[i].stride == 1
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 64, 28, 28)

    # 3 Bottleneck w/ stride 2 and downsample with avg pool
    layer = ResLayer(Bottleneck, 3, 32, 64, stride=2, avg_down=True)
    assert len(layer) == 3
    assert layer[0].in_channels == 32
    assert layer[0].out_channels == 64
    assert layer[0].stride == 2
    assert layer[0].conv1.out_channels == 16
    assert layer[0].downsample is not None and len(layer[0].downsample) == 3
    assert isinstance(layer[0].downsample[0], nn.AvgPool2d)
    assert layer[0].downsample[0].stride == 2
    for i in range(1, 3):
        assert layer[i].in_channels == 64
        assert layer[i].out_channels == 64
        assert layer[i].conv1.out_channels == 16
        assert layer[i].stride == 1
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 64, 28, 28)

    # 3 Bottleneck with custom expansion
    layer = ResLayer(Bottleneck, 3, 32, 32, expansion=2)
    assert len(layer) == 3
    for i in range(3):
        assert layer[i].in_channels == 32
        assert layer[i].out_channels == 32
        assert layer[i].stride == 1
        assert layer[i].conv1.out_channels == 16
        assert layer[i].downsample is None
    x = torch.randn(1, 32, 56, 56)
    x_out = layer(x)
    assert x_out.shape == (1, 32, 56, 56)


def test_resnet():
    """Test resnet backbone."""
    with pytest.raises(KeyError):
        # ResNet depth should be in [18, 34, 50, 101, 152]
        ResNet(20)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=0)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=5)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations) == num_stages
        ResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = ResNet(50)
        model.init_weights(pretrained=0)

    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        ResNet(50, style='tensorflow')

    # Test ResNet50 norm_eval=True
    model = ResNet(50, norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ResNet50 with torchvision pretrained weight
    model = ResNet(depth=50, norm_eval=True)
    model.init_weights('torchvision://resnet50')
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ResNet50 with first stage frozen
    frozen_stages = 1
    model = ResNet(50, frozen_stages=frozen_stages)
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

    # Test ResNet18 forward
    model = ResNet(18, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 64, 56, 56)
    assert feat[1].shape == (1, 128, 28, 28)
    assert feat[2].shape == (1, 256, 14, 14)
    assert feat[3].shape == (1, 512, 7, 7)

    # Test ResNet50 with BatchNorm forward
    model = ResNet(50, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 256, 56, 56)
    assert feat[1].shape == (1, 512, 28, 28)
    assert feat[2].shape == (1, 1024, 14, 14)
    assert feat[3].shape == (1, 2048, 7, 7)

    # Test ResNet50 with layers 1, 2, 3 out forward
    model = ResNet(50, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == (1, 256, 56, 56)
    assert feat[1].shape == (1, 512, 28, 28)
    assert feat[2].shape == (1, 1024, 14, 14)

    # Test ResNet50 with layers 3 (top feature maps) out forward
    model = ResNet(50, out_indices=(3, ))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == (1, 2048, 7, 7)

    # Test ResNet50 with checkpoint forward
    model = ResNet(50, out_indices=(0, 1, 2, 3), with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 256, 56, 56)
    assert feat[1].shape == (1, 512, 28, 28)
    assert feat[2].shape == (1, 1024, 14, 14)
    assert feat[3].shape == (1, 2048, 7, 7)

    # zero initialization of residual blocks
    model = ResNet(50, out_indices=(0, 1, 2, 3), zero_init_residual=True)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert all_zeros(m.norm3)
        elif isinstance(m, BasicBlock):
            assert all_zeros(m.norm2)

    # non-zero initialization of residual blocks
    model = ResNet(50, out_indices=(0, 1, 2, 3), zero_init_residual=False)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert not all_zeros(m.norm3)
        elif isinstance(m, BasicBlock):
            assert not all_zeros(m.norm2)


def test_resnet_v1d():
    model = ResNetV1d(depth=50, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    assert len(model.stem) == 3
    for i in range(3):
        assert isinstance(model.stem[i], ConvModule)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model.stem(imgs)
    assert feat.shape == (1, 64, 112, 112)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 256, 56, 56)
    assert feat[1].shape == (1, 512, 28, 28)
    assert feat[2].shape == (1, 1024, 14, 14)
    assert feat[3].shape == (1, 2048, 7, 7)

    # Test ResNet50V1d with first stage frozen
    frozen_stages = 1
    model = ResNetV1d(depth=50, frozen_stages=frozen_stages)
    assert len(model.stem) == 3
    for i in range(3):
        assert isinstance(model.stem[i], ConvModule)
    model.init_weights()
    model.train()
    check_norm_state(model.stem, False)
    for param in model.stem.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False


def test_resnet_half_channel():
    model = ResNet(50, base_channels=32, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 128, 56, 56)
    assert feat[1].shape == (1, 256, 28, 28)
    assert feat[2].shape == (1, 512, 14, 14)
    assert feat[3].shape == (1, 1024, 7, 7)
