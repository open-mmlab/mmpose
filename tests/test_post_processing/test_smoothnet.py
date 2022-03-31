import numpy as np
import torch

from mmpose.core.post_processing.temporal_filters.builder import build_filter
from mmpose.core.post_processing.temporal_filters import get_pretrained_smoothnet

#  test different data type
def test_data_type_torch():
    noisy_input = torch.randn((100, 17, 3))
    model = get_pretrained_smoothnet(in_length=64)
    # cfg = dict(type='SmoothNet', in_length=64)
    # model = build_filter(cfg)
    out = model(noisy_input)


def test_data_type_torch_zero():
    noisy_input = torch.zeros((50, 20, 3))
    model = get_pretrained_smoothnet(in_length=32)
    # cfg = dict(type='SmoothNet', in_length=32)
    # model = build_filter(cfg)
    out_o = model(noisy_input)


def test_data_type_torch_cuda():
    if not torch.cuda.is_available():
        return
    noisy_input = torch.randn((8, 24, 4)).cuda()
    model = get_pretrained_smoothnet(in_length=8)
    # cfg = dict(type='SmoothNet', in_length=8)
    # model = build_filter(cfg)
    out = model(noisy_input)


def test_data_type_np():
    noisy_input = np.random.rand(100, 24, 6)
    model = get_pretrained_smoothnet(in_length=16)
    # cfg = dict(type='SmoothNet', in_length=16)
    # model = build_filter(cfg)
    out = model(noisy_input)
