# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np

from mmpose.datasets.pipelines import Compose


def _check_flip(origin_imgs, result_imgs):
    """Check if the origin_imgs are flipped correctly."""
    h, w, c = origin_imgs.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                if result_imgs[i, j, k] != origin_imgs[i, w - 1 - j, k]:
                    return False
    return True


def get_sample_data():
    ann_info = {}
    ann_info['image_size'] = np.array([256, 256])
    ann_info['heatmap_size'] = np.array([64, 64, 64])
    ann_info['heatmap3d_depth_bound'] = 400.0
    ann_info['heatmap_size_root'] = 64
    ann_info['root_depth_bound'] = 400.0
    ann_info['num_joints'] = 42
    ann_info['joint_weights'] = np.ones((ann_info['num_joints'], 1),
                                        dtype=np.float32)
    ann_info['use_different_joint_weights'] = False
    ann_info['flip_pairs'] = [[i, 21 + i] for i in range(21)]
    ann_info['inference_channel'] = list(range(42))
    ann_info['num_output_channels'] = 42
    ann_info['dataset_channel'] = list(range(42))

    results = {
        'image_file': 'tests/data/interhand2.6m/image69148.jpg',
        'center': np.asarray([200, 200], dtype=np.float32),
        'scale': 1.0,
        'rotation': 0,
        'joints_3d': np.zeros([42, 3], dtype=np.float32),
        'joints_3d_visible': np.ones([42, 3], dtype=np.float32),
        'hand_type': np.asarray([1, 0], dtype=np.float32),
        'hand_type_valid': 1,
        'rel_root_depth': 50.0,
        'rel_root_valid': 1,
        'ann_info': ann_info
    }
    return results


def test_hand_transforms():
    results = get_sample_data()

    # load image
    pipeline = Compose([dict(type='LoadImageFromFile')])
    results = pipeline(results)

    # test random flip
    pipeline = Compose([dict(type='HandRandomFlip', flip_prob=1)])
    results_flip = pipeline(copy.deepcopy(results))
    assert _check_flip(results['img'], results_flip['img'])

    # test root depth target generation
    pipeline = Compose([dict(type='HandGenerateRelDepthTarget')])
    results_depth = pipeline(copy.deepcopy(results))
    assert results_depth['target'].shape == (1, )
    assert results_depth['target_weight'].shape == (1, )
