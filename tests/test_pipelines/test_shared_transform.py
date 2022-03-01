# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
from mmcv import bgr2rgb, build_from_cfg

from mmpose.datasets import PIPELINES
from mmpose.datasets.pipelines import Compose


def check_keys_equal(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys) == set(result_keys)


def check_keys_contain(result_keys, target_keys):
    """Check if elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def test_compose():
    with pytest.raises(TypeError):
        # transform must be callable or a dict
        Compose('LoadImageFromFile')

    target_keys = ['img', 'img_rename', 'img_metas']

    # test Compose given a data pipeline
    img = np.random.randn(256, 256, 3)
    results = dict(img=img, img_file='test_image.png')
    test_pipeline = [
        dict(
            type='Collect',
            keys=['img', ('img', 'img_rename')],
            meta_keys=['img_file'])
    ]
    compose = Compose(test_pipeline)
    compose_results = compose(results)
    assert check_keys_equal(compose_results.keys(), target_keys)
    assert check_keys_equal(compose_results['img_metas'].data.keys(),
                            ['img_file'])

    # test Compose when forward data is None
    results = None

    class ExamplePipeline:

        def __call__(self, results):
            return None

    nonePipeline = ExamplePipeline()
    test_pipeline = [nonePipeline]
    compose = Compose(test_pipeline)
    compose_results = compose(results)
    assert compose_results is None

    assert repr(compose) == compose.__class__.__name__ + \
        f'(\n    {nonePipeline}\n)'


def test_load_image_from_file():
    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    data_prefix = 'tests/data/coco/'
    image_file = osp.join(data_prefix, '00000000078.jpg')
    results = dict(image_file=image_file)

    # load an image that doesn't exist
    with pytest.raises(FileNotFoundError):
        results = load(results)

    # mormal loading
    image_file = osp.join(data_prefix, '000000000785.jpg')
    results = dict(image_file=image_file)
    results = load(results)
    assert results['img'].shape == (425, 640, 3)

    # load a single image from a list
    image_file = [osp.join(data_prefix, '000000000785.jpg')]
    results = dict(image_file=image_file)
    results = load(results)
    assert len(results['img']) == 1

    # test loading multi images from a list
    image_file = [
        osp.join(data_prefix, '000000000785.jpg'),
        osp.join(data_prefix, '00000004008.jpg'),
    ]
    results = dict(image_file=image_file)

    with pytest.raises(FileNotFoundError):
        results = load(results)

    image_file = [
        osp.join(data_prefix, '000000000785.jpg'),
        osp.join(data_prefix, '000000040083.jpg'),
    ]
    results = dict(image_file=image_file)

    results = load(results)
    assert len(results['img']) == 2

    # manually set image outside the pipeline
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    results = load(dict(img=img))
    np.testing.assert_equal(results['img'], bgr2rgb(img))

    imgs = np.random.randint(0, 255, (2, 32, 32, 3), dtype=np.uint8)
    desired = np.concatenate([bgr2rgb(img) for img in imgs], axis=0)
    results = load(dict(img=imgs))
    np.testing.assert_equal(results['img'], desired)

    # neither 'image_file' or valid 'img' is given
    results = dict()
    with pytest.raises(KeyError):
        _ = load(results)

    results = dict(img=np.random.randint(0, 255, (32, 32), dtype=np.uint8))
    with pytest.raises(ValueError):
        _ = load(results)


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


def test_multitask_gather():
    ann_info = dict(
        image_size=np.array([256, 256]),
        heatmap_size=np.array([64, 64]),
        num_joints=17,
        joint_weights=np.ones((17, 1), dtype=np.float32),
        use_different_joint_weights=False)

    results = dict(
        joints_3d=np.zeros([17, 3]),
        joints_3d_visible=np.ones([17, 3]),
        ann_info=ann_info)

    pipeline_list = [[dict(type='TopDownGenerateTarget', sigma=2)],
                     [dict(type='TopDownGenerateTargetRegression')]]
    pipeline = dict(
        type='MultitaskGatherTarget',
        pipeline_list=pipeline_list,
        pipeline_indices=[0, 1, 0],
    )
    pipeline = build_from_cfg(pipeline, PIPELINES)

    results = pipeline(results)
    target = results['target']
    target_weight = results['target_weight']
    assert isinstance(target, list)
    assert isinstance(target_weight, list)
    assert target[0].shape == (17, 64, 64)
    assert target_weight[0].shape == (17, 1)
    assert target[1].shape == (17, 2)
    assert target_weight[1].shape == (17, 2)
    assert target[2].shape == (17, 64, 64)
    assert target_weight[2].shape == (17, 1)


def test_rename_keys():
    results = dict(
        joints_3d=np.ones([17, 3]), joints_3d_visible=np.ones([17, 3]))
    pipeline = dict(
        type='RenameKeys',
        key_pairs=[('joints_3d', 'target'),
                   ('joints_3d_visible', 'target_weight')])
    pipeline = build_from_cfg(pipeline, PIPELINES)
    results = pipeline(results)
    assert 'joints_3d' not in results
    assert 'joints_3d_visible' not in results
    assert 'target' in results
    assert 'target_weight' in results
    assert results['target'].shape == (17, 3)
    assert results['target_weight'].shape == (17, 3)
