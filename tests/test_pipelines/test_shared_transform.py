import os.path as osp

import numpy as np
from mmcv.utils import build_from_cfg

from mmpose.datasets import PIPELINES


def test_albu_transform():
    data_prefix = 'tests/data/coco/'
    results = dict(image_file=osp.join(data_prefix, '000000000785.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    albu_transform = dict(
        type='Albumentation',
        transforms=[
            dict(type='RandomBrightnessContrast', p=0.2),
            dict(type='ToFloat')
        ])
    albu_transform = build_from_cfg(albu_transform, PIPELINES)

    # Execute transforms
    results = load(results)

    results = albu_transform(results)

    assert results['img'].dtype == np.float32


def test_photometric_distortion_transform():
    data_prefix = 'tests/data/coco/'
    results = dict(image_file=osp.join(data_prefix, '000000000785.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    photo_transform = dict(type='PhotometricDistortion')
    photo_transform = build_from_cfg(photo_transform, PIPELINES)

    # Execute transforms
    results = load(results)

    results = photo_transform(results)

    assert results['img'].dtype == np.uint8
