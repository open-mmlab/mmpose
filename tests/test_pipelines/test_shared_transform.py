import copy
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import build_from_cfg
from torchvision import transforms

from mmpose.datasets import PIPELINES


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)

    # test repr
    transform = dict(type='Normalize', **img_norm_cfg)
    normalize_module = build_from_cfg(transform, PIPELINES)
    assert isinstance(repr(normalize_module), str)

    # read data
    results = dict()
    img = mmcv.imread('tests/data/coco/000000000785.jpg', 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    norm_results = normalize_module(results)
    assert np.equal(norm_results['img'], norm_results['img2']).all()

    # compare results with manual computation
    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    normalized_img = (original_img[..., ::-1] - mean) / std
    assert np.allclose(norm_results['img'], normalized_img)

    # compare results with torchvision
    normalize_module = transforms.Normalize(mean=mean, std=std)
    tensor_img = original_img[..., ::-1].copy()
    tensor_img = torch.Tensor(tensor_img.transpose(2, 0, 1))
    normalized_img = normalize_module(tensor_img)
    normalized_img = np.array(normalized_img).transpose(1, 2, 0)
    assert np.equal(norm_results['img'], normalized_img).all()


def test_albu_transform():
    data_prefix = 'tests/data/coco/'
    results = dict(image_file=osp.join(data_prefix, '000000000785.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    albu_transform = dict(
        type='Albu',
        transforms=[
            dict(type='RandomBrightnessContrast', p=0.2),
            dict(type='ToFloat')
        ])
    albu_transform = build_from_cfg(albu_transform, PIPELINES)

    # Execute transforms
    results = load(results)

    results = albu_transform(results)

    assert results['img'].dtype == np.float32
