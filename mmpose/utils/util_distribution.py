# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.utils import (IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE)

dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'cuda': MMDistributedDataParallel}


def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel model; if device is npu,
    return a NPUDataParallel model.

    Args:
        model (:class:`nn.Module`): model to be parallelized.
        device (str): device type, cuda, cpu or npu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        nn.Module: the model to be parallelized.
    """
    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        dp_factory['npu'] = NPUDataParallel
        torch.npu.set_device(kwargs['device_ids'][0])
        torch.npu.set_compile_mode(jit_compile=False)
        model = model.npu()
    elif device == 'cuda':
        model = model.cuda(kwargs['device_ids'][0])

    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel model;
    if device is npu, return a NPUDistributedDataParallel model.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, npu or cuda.

    Returns:
        :class:`nn.Module`: the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'npu'], \
        'Only available for cuda or npu devices.'
    if device == 'npu':
        from mmcv.device.npu import NPUDistributedDataParallel
        torch.npu.set_compile_mode(jit_compile=False)
        ddp_factory['npu'] = NPUDistributedDataParallel
        model = model.npu()
    elif device == 'cuda':
        model = model.cuda()

    return ddp_factory[device](model, *args, **kwargs)

def get_device() -> str:
    """Returns the currently existing device type.

    .. note::
        Since npu provides tools to automatically convert cuda functions,
        we need to make judgments on npu first to avoid entering
        the cuda branch when using npu.

    Returns:
        str: cuda | npu | cpu.
    """
    if IS_NPU_AVAILABLE:
        return 'npu'
    elif IS_CUDA_AVAILABLE:
        return 'cuda'
    else:
        return 'cpu'
