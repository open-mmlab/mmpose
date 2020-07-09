import copy
import os.path as osp

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from pycocotools.coco import COCO

from mmpose.datasets.pipelines import (AffineTransform, Collect,
                                       GenerateTarget, HalfBodyTransform,
                                       LoadImageFromFile, NormalizeTensor,
                                       RandomFlip, ToTensor)


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


def test_pipeline():
    # test loading
    data_prefix = 'tests/data/'
    ann_file = osp.join(data_prefix, 'test_coco.json')
    coco = COCO(ann_file)

    results = dict(image_file=osp.join(data_prefix, '000000000785.jpg'))
    transform = LoadImageFromFile()
    results = transform(copy.deepcopy(results))
    assert results['image_file'] == osp.join(data_prefix, '000000000785.jpg')

    assert results['img'].shape == (425, 640, 3)
    image_size = (425, 640)

    ann_ids = coco.getAnnIds(785)
    ann = coco.anns[ann_ids[0]]

    num_joints = 17
    joints_3d = np.zeros((num_joints, 3), dtype=np.float)
    joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float)
    for ipt in range(num_joints):
        joints_3d[ipt, 0] = ann['keypoints'][ipt * 3 + 0]
        joints_3d[ipt, 1] = ann['keypoints'][ipt * 3 + 1]
        joints_3d[ipt, 2] = 0
        t_vis = ann['keypoints'][ipt * 3 + 2]
        if t_vis > 1:
            t_vis = 1
        joints_3d_visible[ipt, 0] = t_vis
        joints_3d_visible[ipt, 1] = t_vis
        joints_3d_visible[ipt, 2] = 0

    center, scale = _box2cs(ann['bbox'][:4], image_size)

    results['joints_3d'] = joints_3d
    results['joints_3d_visible'] = joints_3d_visible
    results['center'] = center
    results['scale'] = scale
    results['bbox_score'] = 1

    results['ann_info'] = {}
    results['ann_info']['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8],
                                         [9, 10], [11, 12], [13, 14], [15, 16]]
    results['ann_info']['num_joints'] = num_joints
    results['ann_info']['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    results['ann_info']['lower_body_ids'] = (11, 12, 13, 14, 15, 16)
    results['ann_info']['use_different_joints_weight'] = False
    results['ann_info']['joints_weight'] = np.array([
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
                                                    dtype=np.float32).reshape(
                                                        (num_joints, 1))
    results['ann_info']['image_size'] = np.array([192, 256])
    results['ann_info']['heatmap_size'] = np.array([48, 64])

    # test filp
    random_flip = RandomFlip(flip_prob=1.)
    results_flip = random_flip(copy.deepcopy(results))
    assert _check_flip(results['img'], results_flip['img'])

    # test halfbody transform
    halfbody_transform = HalfBodyTransform(
        num_joints_half_body=8, prob_half_body=1.)
    results_halfbody = halfbody_transform(copy.deepcopy(results))
    assert (results_halfbody['scale'] <= results['scale']).all()

    affine_transform = AffineTransform()
    results['rotation'] = 90
    results_affine = affine_transform(copy.deepcopy(results))
    assert results_affine['img'].shape == (256, 192, 3)

    results = results_affine
    to_tensor = ToTensor()
    results_tensor = to_tensor(copy.deepcopy(results))
    assert isinstance(results_tensor['img'], torch.Tensor)
    assert results_tensor['img'].shape == torch.Size([3, 256, 192])

    norm_cfg = {}
    norm_cfg['mean'] = [0.485, 0.456, 0.406]
    norm_cfg['std'] = [0.229, 0.224, 0.225]

    normalize = NormalizeTensor(mean=norm_cfg['mean'], std=norm_cfg['std'])

    results_normalize = normalize(copy.deepcopy(results_tensor))
    _check_normalize(results_tensor['img'].data.numpy(),
                     results_normalize['img'].data.numpy(), norm_cfg)

    generate_target = GenerateTarget(sigma=2, unbiased_encoding=False)
    results_target = generate_target(copy.deepcopy(results_tensor))
    assert 'target' in results_target
    assert results_target['target'].shape == (
        num_joints, results['ann_info']['heatmap_size'][1],
        results['ann_info']['heatmap_size'][0])
    assert 'target_weight' in results_target
    assert results_target['target_weight'].shape == (num_joints, 1)

    collect = Collect(
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ])
    results_final = collect(results_target)
    assert 'img_size' not in results_final['img_metas'].data
    assert 'image_file' in results_final['img_metas'].data
