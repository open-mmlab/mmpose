# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

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
