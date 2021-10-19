# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from mmpose.datasets.pipelines import (Collect, IUVToTensor, LoadImageFromFile,
                                       LoadIUVFromFile, MeshAffine,
                                       MeshGetRandomScaleRotation,
                                       MeshRandomChannelNoise, MeshRandomFlip,
                                       NormalizeTensor, ToTensor)


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


def _load_test_data():
    data_cfg = dict(
        image_size=[256, 256],
        iuv_size=[64, 64],
        num_joints=24,
        use_IUV=True,
        uv_type='BF')
    ann_file = 'tests/data/h36m/test_h36m.npz'
    img_prefix = 'tests/data/h36m'
    index = 0

    ann_info = dict(image_size=np.array(data_cfg['image_size']))
    ann_info['iuv_size'] = np.array(data_cfg['iuv_size'])
    ann_info['num_joints'] = data_cfg['num_joints']
    ann_info['flip_pairs'] = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                              [20, 21], [22, 23]]
    ann_info['use_different_joint_weights'] = False
    ann_info['joint_weights'] = \
        np.ones(ann_info['num_joints'], dtype=np.float32
                ).reshape(ann_info['num_joints'], 1)
    ann_info['uv_type'] = data_cfg['uv_type']
    ann_info['use_IUV'] = data_cfg['use_IUV']
    uv_type = ann_info['uv_type']
    iuv_prefix = os.path.join(img_prefix, f'{uv_type}_IUV_gt')

    ann_data = np.load(ann_file)

    results = dict(ann_info=ann_info)
    results['rotation'] = 0
    results['image_file'] = os.path.join(img_prefix,
                                         ann_data['imgname'][index])
    scale = ann_data['scale'][index]
    results['scale'] = np.array([scale, scale]).astype(np.float32)
    results['center'] = ann_data['center'][index].astype(np.float32)

    # Get gt 2D joints, if available
    if 'part' in ann_data.keys():
        keypoints = ann_data['part'][index].astype(np.float32)
        results['joints_2d'] = keypoints[:, :2]
        results['joints_2d_visible'] = keypoints[:, -1][:, np.newaxis]
    else:
        results['joints_2d'] = np.zeros((24, 2), dtype=np.float32)
        results['joints_2d_visible'] = np.zeros((24, 1), dtype=np.float32)

    # Get gt 3D joints, if available
    if 'S' in ann_data.keys():
        joints_3d = ann_data['S'][index].astype(np.float32)
        results['joints_3d'] = joints_3d[:, :3]
        results['joints_3d_visible'] = joints_3d[:, -1][:, np.newaxis]
    else:
        results['joints_3d'] = np.zeros((24, 3), dtype=np.float32)
        results['joints_3d_visible'] = np.zeros((24, 1), dtype=np.float32)

    # Get gt SMPL parameters, if available
    if 'pose' in ann_data.keys() and 'shape' in ann_data.keys():
        results['pose'] = ann_data['pose'][index].astype(np.float32)
        results['beta'] = ann_data['shape'][index].astype(np.float32)
        results['has_smpl'] = 1
    else:
        results['pose'] = np.zeros(72, dtype=np.float32)
        results['beta'] = np.zeros(10, dtype=np.float32)
        results['has_smpl'] = 0

    # Get gender data, if available
    if 'gender' in ann_data.keys():
        gender = ann_data['gender'][index]
        results['gender'] = 0 if str(gender) == 'm' else 1
    else:
        results['gender'] = -1

    # Get IUV image, if available
    if 'iuv_names' in ann_data.keys():
        results['iuv_file'] = os.path.join(iuv_prefix,
                                           ann_data['iuv_names'][index])
        results['has_iuv'] = results['has_smpl']
    else:
        results['iuv_file'] = ''
        results['has_iuv'] = 0

    return copy.deepcopy(results)


def test_mesh_pipeline():
    # load data
    results = _load_test_data()

    # data_prefix = 'tests/data/coco/'
    # ann_file = osp.join(data_prefix, 'test_coco.json')
    # coco = COCO(ann_file)
    #
    # results = dict(image_file=osp.join(data_prefix, '000000000785.jpg'))

    # test loading image
    transform = LoadImageFromFile()
    results = transform(copy.deepcopy(results))
    assert results['img'].shape == (1002, 1000, 3)

    # test loading densepose IUV image without GT iuv image
    transform = LoadIUVFromFile()
    results_no_iuv = copy.deepcopy(results)
    results_no_iuv['has_iuv'] = 0
    results_no_iuv = transform(results_no_iuv)
    assert results_no_iuv['iuv'] is None

    # test loading densepose IUV image
    results = transform(results)
    assert results['iuv'].shape == (1002, 1000, 3)
    assert results['iuv'][:, :, 0].max() <= 1

    # test flip
    random_flip = MeshRandomFlip(flip_prob=1.)
    results_flip = random_flip(copy.deepcopy(results))
    assert _check_flip(results['img'], results_flip['img'])
    flip_iuv = results_flip['iuv']
    flip_iuv[:, :, 1] = 255 - flip_iuv[:, :, 1]
    assert _check_flip(results['iuv'], flip_iuv)
    results = results_flip

    # test flip without IUV image
    results_no_iuv = random_flip(copy.deepcopy(results_no_iuv))
    assert results_no_iuv['iuv'] is None

    # test random scale and rotation
    random_scale_rotation = MeshGetRandomScaleRotation()
    results = random_scale_rotation(results)

    # test affine
    affine_transform = MeshAffine()
    results_affine = affine_transform(copy.deepcopy(results))
    assert results_affine['img'].shape == (256, 256, 3)
    assert results_affine['iuv'].shape == (64, 64, 3)
    results = results_affine

    # test affine without IUV image
    results_no_iuv['rotation'] = 30
    results_no_iuv = affine_transform(copy.deepcopy(results_no_iuv))
    assert results_no_iuv['iuv'] is None

    # test channel noise
    random_noise = MeshRandomChannelNoise()
    results_noise = random_noise(copy.deepcopy(results))
    results = results_noise

    # transfer image to tensor
    to_tensor = ToTensor()
    results_tensor = to_tensor(copy.deepcopy(results))
    assert isinstance(results_tensor['img'], torch.Tensor)
    assert results_tensor['img'].shape == torch.Size([3, 256, 256])

    # transfer IUV image to tensor
    iuv_to_tensor = IUVToTensor()
    results_tensor = iuv_to_tensor(results_tensor)
    assert isinstance(results_tensor['part_index'], torch.LongTensor)
    assert results_tensor['part_index'].shape == torch.Size([1, 64, 64])
    max_I = results_tensor['part_index'].max().item()
    assert (max_I == 0 or max_I == 1)
    assert isinstance(results_tensor['uv_coordinates'], torch.FloatTensor)
    assert results_tensor['uv_coordinates'].shape == torch.Size([2, 64, 64])

    # transfer IUV image to tensor without GT IUV image
    results_no_iuv = iuv_to_tensor(results_no_iuv)
    assert isinstance(results_no_iuv['part_index'], torch.LongTensor)
    assert results_no_iuv['part_index'].shape == torch.Size([1, 64, 64])
    max_I = results_no_iuv['part_index'].max().item()
    assert (max_I == 0)
    assert isinstance(results_no_iuv['uv_coordinates'], torch.FloatTensor)
    assert results_no_iuv['uv_coordinates'].shape == torch.Size([2, 64, 64])

    # test norm
    norm_cfg = {}
    norm_cfg['mean'] = [0.485, 0.456, 0.406]
    norm_cfg['std'] = [0.229, 0.224, 0.225]
    normalize = NormalizeTensor(mean=norm_cfg['mean'], std=norm_cfg['std'])

    results_normalize = normalize(copy.deepcopy(results_tensor))
    _check_normalize(results_tensor['img'].data.numpy(),
                     results_normalize['img'].data.numpy(), norm_cfg)

    # test collect
    collect = Collect(
        keys=[
            'img', 'joints_2d', 'joints_2d_visible', 'joints_3d',
            'joints_3d_visible', 'pose', 'beta', 'part_index', 'uv_coordinates'
        ],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'iuv_file'])
    results_final = collect(results_normalize)

    assert 'img_size' not in results_final['img_metas'].data
    assert 'image_file' in results_final['img_metas'].data
