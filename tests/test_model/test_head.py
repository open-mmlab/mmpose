import numpy as np
import pytest
import torch

from mmpose.models import MultiStageHead, SimpleHead


def test_simple_head():
    """Test simple head."""
    with pytest.raises(TypeError):
        # extra
        _ = SimpleHead(out_channels=3, in_channels=512, extra=[])

    # test num deconv layers
    with pytest.raises(ValueError):
        _ = SimpleHead(out_channels=3, in_channels=512, num_deconv_layers=-1)

    _ = SimpleHead(out_channels=3, in_channels=512, num_deconv_layers=0)

    with pytest.raises(ValueError):
        # the number of layers should match
        _ = SimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4))

    with pytest.raises(ValueError):
        # the number of kernels should match
        _ = SimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = SimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(3, 2, 0))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = SimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, -1))

    # test final_conv_kernel
    head = SimpleHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 3})
    head.init_weights()
    assert head.final_layer.padding == (1, 1)
    head = SimpleHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 1})
    assert head.final_layer.padding == (0, 0)
    with pytest.raises(AssertionError):
        # the deconv kernels should be 4, 3, 2
        _ = SimpleHead(
            out_channels=3, in_channels=512, extra={'final_conv_kernel': 0})

    head = SimpleHead(out_channels=3, in_channels=512)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 3, 256, 256])

    head = SimpleHead(out_channels=3, in_channels=512, num_deconv_layers=0)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 3, 32, 32])

    head = SimpleHead(out_channels=3, in_channels=512, num_deconv_layers=0)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out.shape == torch.Size([1, 3, 32, 32])

    head.init_weights()


def test_multistage_head():
    """Test multistage head."""
    with pytest.raises(TypeError):
        # the number of layers should match
        _ = MultiStageHead(
            out_channels=3, in_channels=512, num_stages=1, extra=[])

    # test num deconv layers
    with pytest.raises(ValueError):
        _ = MultiStageHead(
            out_channels=3, in_channels=512, num_deconv_layers=-1)

    _ = MultiStageHead(out_channels=3, in_channels=512, num_deconv_layers=0)

    with pytest.raises(ValueError):
        # the number of layers should match
        _ = MultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4))

    with pytest.raises(ValueError):
        # the number of kernels should match
        _ = MultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = MultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(3, 2, 0))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = MultiStageHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, -1))

    with pytest.raises(AssertionError):
        # inputs should be list
        head = MultiStageHead(out_channels=3, in_channels=512)
        input_shape = (1, 512, 32, 32)
        inputs = _demo_inputs(input_shape)
        out = head(inputs)

    # test final_conv_kernel
    head = MultiStageHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 3})
    head.init_weights()
    assert head.multi_final_layers[0].padding == (1, 1)
    head = MultiStageHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 1})
    assert head.multi_final_layers[0].padding == (0, 0)
    with pytest.raises(AssertionError):
        # the deconv kernels should be 4, 3, 2
        _ = MultiStageHead(
            out_channels=3, in_channels=512, extra={'final_conv_kernel': 0})

    head = MultiStageHead(out_channels=3, in_channels=512)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert len(out) == 1
    assert out[0].shape == torch.Size([1, 3, 256, 256])

    head = MultiStageHead(out_channels=3, in_channels=512, num_deconv_layers=0)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out[0].shape == torch.Size([1, 3, 32, 32])

    head.init_weights()


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
