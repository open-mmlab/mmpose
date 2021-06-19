import torch.nn as nn
import torch
import numpy as np
import math


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp[0], out[0])
    elif type(module).__name__ == 'ConvFunction':
        return compute_Conv2d_flops(module, inp[0], out[0])
    elif type(module).__name__ == 'SplitKernelConvFunction':
        return compute_Conv2d_flops(module, inp[0], out[0])
    elif isinstance(module, nn.ConvTranspose2d):
        return compute_ConvTranspose2d_flops(module, inp[0], out[0])
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_flops(module, inp[0], out[0])
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_flops(module, inp[0], out[0])
    elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        return compute_adaptivepool_flops(module, inp[0], out[0])
    elif isinstance(module,
                    (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_flops(module, inp[0], out[0])
    elif isinstance(module, nn.Upsample):
        return compute_Upsample_flops(module, inp[0], out[0])
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp[0], out[0])
    elif type(module).__name__ == 'MatMul':
        return compute_matmul_flops(module, inp, out)
    else:
        #print(f"[Flops]: {type(module).__name__} is not supported!")
        return 0
    pass


def compute_matmul_flops(moudle, inp, out):
    x, y = inp
    batch_size = x.size(0)
    _, l, m = x.size()
    _, _, n = y.size()
    return batch_size * 2 * l * m * n


def compute_Conv2d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    # assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_ConvTranspose2d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.ConvTranspose2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_h, in_w = inp.size()[2:]

    k_h, k_w = module.kernel_size
    in_c = module.in_channels
    out_c = module.out_channels
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * in_h * in_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        out_h, out_w = out.size()[2:]
        bias_flops = out_c * batch_size * out_h * out_w

    total_flops = total_conv_flops + bias_flops

    return total_flops


def compute_adaptivepool_flops(module, input, output):
    # credits: https://github.com/xternalz/SDPoint/blob/master/utils/flops.py
    batch_size = input.size(0)
    input_planes = input.size(1)
    input_height = input.size(2)
    input_width = input.size(3)

    flops = 0
    for i in range(output.size(2)):
        y_start = int(math.floor(float(i * input_height) / output.size(2)))
        y_end = int(math.ceil(float((i + 1) * input_height) / output.size(2)))
        for j in range(output.size(3)):
            x_start = int(math.floor(float(j * input_width) / output.size(3)))
            x_end = int(
                math.ceil(float((j + 1) * input_width) / output.size(3)))

            flops += batch_size * input_planes * (y_end - y_start + 1) * (
                x_end - x_start + 1)
    return flops


def compute_BatchNorm2d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    in_c, in_h, in_w = inp.size()[1:]
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops


def compute_ReLU_flops(module, inp, out):
    assert isinstance(module,
                      (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    active_elements_count = batch_size

    for s in inp.size()[1:]:
        active_elements_count *= s

    return active_elements_count


def compute_Pool2d_flops(module, input, out):
    batch_size = input.size(0)
    input_planes = input.size(1)
    input_height = input.size(2)
    input_width = input.size(3)
    kernel_size = ('int' in str(type(module.kernel_size))) and [
        module.kernel_size, module.kernel_size
    ] or module.kernel_size
    kernel_ops = kernel_size[0] * kernel_size[1]
    stride = ('int' in str(type(
        module.stride))) and [module.stride, module.stride] or module.stride
    padding = ('int' in str(type(module.padding))) and [
        module.padding, module.padding
    ] or module.padding

    output_width = math.floor((input_width + 2 * padding[0] - kernel_size[0]) /
                              float(stride[0]) + 1)
    output_height = math.floor(
        (input_height + 2 * padding[1] - kernel_size[1]) / float(stride[0]) +
        1)
    return batch_size * input_planes * output_width * output_height * kernel_ops


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    assert len(inp.size()) == 2 and len(out.size()) == 2
    batch_size = inp.size()[0]
    return batch_size * inp.size()[1] * out.size()[1]


def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    for s in output_size.shape[1:]:
        output_elements_count *= s

    return output_elements_count
