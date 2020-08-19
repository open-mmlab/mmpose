import copy

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from mmpose.datasets import DATASETS
from mmpose.datasets.pipelines import (Collect, IUVToTensor, LoadImageFromFile,
                                       LoadIUVFromFile, MeshAffine,
                                       MeshRandomFlip, NormalizeTensor,
                                       ToTensor)


def _check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def _check_flip(origin_imgs, result_imgs):
    """Check if the origin_imgs are flipped correctly."""
    h, w, c = origin_imgs.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                if result_imgs[i, j, k] != origin_imgs[i, w - 1 - j, k]:
                    return False
    return True


def _check_rot90(origin_imgs, result_imgs):
    if origin_imgs.shape[0] == result_imgs.shape[1] and \
            origin_imgs.shape[1] == result_imgs.shape[0]:
        return True
    else:
        return False


def _check_normalize(origin_imgs, result_imgs, norm_cfg):
    """Check if the origin_imgs are normalized correctly into result_imgs in a
    given norm_cfg."""
    target_imgs = result_imgs.copy()
    for i in range(3):
        target_imgs[i] *= norm_cfg['std'][i]
        target_imgs[i] += norm_cfg['mean'][i]
    assert_array_almost_equal(origin_imgs, target_imgs, decimal=4)


def _box2cs(box, image_size):
    x, y, w, h = box[:4]

    aspect_ratio = 1. * image_size[0] / image_size[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200.0, h * 1.0 / 200.0], dtype=np.float32)
    scale = scale * 1.25
    return center, scale


def test_mesh_pipeline():
    # test loading
    dataset = 'MeshH36MDataset'
    dataset_class = DATASETS.get(dataset)
    data_cfg = dict(
        image_size=[256, 256],
        iuv_size=[64, 64],
        num_joints=24,
        use_IUV=True,
        uv_type='BF')
    h36m_dataset = dataset_class(
        ann_file='tests/data/h36m/test_h36m.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=False)

    results = h36m_dataset[0]
    transform = LoadImageFromFile()
    results = transform(copy.deepcopy(results))
    assert results['img'].shape == (1002, 1000, 3)

    transform = LoadIUVFromFile()
    results = transform(results)
    assert results['iuv'].shape == (1002, 1000, 3)
    assert results['iuv'][:, :, 0].max() <= 1

    # test filp
    random_flip = MeshRandomFlip(flip_prob=1.)
    results_flip = random_flip(copy.deepcopy(results))
    assert _check_flip(results['img'], results_flip['img'])
    flip_iuv = results_flip['iuv']
    flip_iuv[:, :, 1] = 255 - flip_iuv[:, :, 1]
    assert _check_flip(results['iuv'], flip_iuv)

    affine_transform = MeshAffine()
    results['rotation'] = 90
    results_affine = affine_transform(copy.deepcopy(results))
    assert results_affine['img'].shape == (256, 256, 3)
    assert results_affine['iuv'].shape == (256, 256, 3)

    results = results_affine
    to_tensor = ToTensor()
    results_tensor = to_tensor(copy.deepcopy(results))
    assert isinstance(results_tensor['img'], torch.Tensor)
    assert results_tensor['img'].shape == torch.Size([3, 256, 256])

    iuv_to_tensor = IUVToTensor()
    results_tensor = iuv_to_tensor(results_tensor)
    assert isinstance(results_tensor['iuv'], torch.Tensor)
    assert results_tensor['iuv'].shape == torch.Size([3, 256, 256])
    max_I = results_tensor['iuv'][:, :, 0].max().item()
    assert (max_I == 0 or max_I == 1)

    norm_cfg = {}
    norm_cfg['mean'] = [0.485, 0.456, 0.406]
    norm_cfg['std'] = [0.229, 0.224, 0.225]
    normalize = NormalizeTensor(mean=norm_cfg['mean'], std=norm_cfg['std'])

    results_normalize = normalize(copy.deepcopy(results_tensor))
    _check_normalize(results_tensor['img'].data.numpy(),
                     results_normalize['img'].data.numpy(), norm_cfg)

    collect = Collect(
        keys=[
            'img', 'joints_2d', 'joints_2d_visible', 'joints_3d',
            'joints_3d_visible', 'pose', 'beta', 'iuv'
        ],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'iuv_file'])
    results_final = collect(results_normalize)

    assert 'img_size' not in results_final['img_metas'].data
    assert 'image_file' in results_final['img_metas'].data
