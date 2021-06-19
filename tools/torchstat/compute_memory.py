import torch.nn as nn
import torch
import numpy as np


def compute_memory(module, inp, out):
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_memory(module, inp[0], out[0])
    elif isinstance(module, nn.PReLU):
        return compute_PReLU_memory(module, inp[0], out[0])
    elif isinstance(module, nn.Conv2d):
        return compute_Conv2d_memory(module, inp[0], out[0])
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_memory(module, inp[0], out[0])
    elif isinstance(module, nn.Linear):
        return compute_Linear_memory(module, inp[0], out[0])
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_memory(module, inp[0], out[0])
    else:
        #print(f"[Memory]: {type(module).__name__} is not supported!")
        return 0, 0
    pass


def num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_ReLU_memory(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU))

    mread = inp.numel()
    mwrite = out.numel()

    return mread * inp.element_size(), mwrite * out.element_size()


def compute_PReLU_memory(module, inp, out):
    assert isinstance(module, nn.PReLU)

    batch_size = inp.size()[0]
    mread = batch_size * (inp[0].numel() + num_params(module))
    mwrite = out.numel()

    return mread * inp.element_size(), mwrite * out.element_size()


def compute_Conv2d_memory(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]

    # This includes weights with bias if the module contains it.
    mread = batch_size * (inp[0].numel() + num_params(module))
    mwrite = out.numel()

    return mread * inp.element_size(), mwrite * out.element_size()


def compute_BatchNorm2d_memory(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size, in_c, in_h, in_w = inp.size()
    mread = batch_size * (inp[0].numel() + 2 * in_c)
    mwrite = out.numel()

    return mread * inp.element_size(), mwrite * out.element_size()


def compute_Linear_memory(module, inp, out):
    assert isinstance(module, nn.Linear)
    assert len(inp.size()) == 2 and len(out.size()) == 2

    batch_size = inp.size()[0]

    # This includes weights with bias if the module contains it.
    mread = batch_size * (inp[0].numel() + num_params(module))
    mwrite = out.numel()

    return mread * inp.element_size(), mwrite * out.element_size()


def compute_Pool2d_memory(module, inp, out):
    assert isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    mread = inp.numel()
    mwrite = out.numel()

    return mread * inp.element_size(), mwrite * out.element_size()
