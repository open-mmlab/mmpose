# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from xtcocotools.coco import COCO

from mmpose.datasets.pipelines import (Collect, LoadImageFromFile,
                                       NormalizeTensor, TopDownAffine,
                                       TopDownGenerateTarget,
                                       TopDownGetBboxCenterScale,
                                       TopDownGetRandomScaleRotation,
                                       TopDownHalfBodyTransform,
                                       TopDownRandomFlip,
                                       TopDownRandomShiftBboxCenter, ToTensor)


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


def test_top_down_pipeline():
    # test loading
    data_prefix = 'tests/data/coco/'
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
    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
    joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
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
    results['bbox_id'] = 0

    results['ann_info'] = {}
    results['ann_info']['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8],
                                         [9, 10], [11, 12], [13, 14], [15, 16]]
    results['ann_info']['num_joints'] = num_joints
    results['ann_info']['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    results['ann_info']['lower_body_ids'] = (11, 12, 13, 14, 15, 16)
    results['ann_info']['use_different_joint_weights'] = False
    results['ann_info']['joint_weights'] = np.array([
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
                                                    dtype=np.float32).reshape(
                                                        (num_joints, 1))
    results['ann_info']['image_size'] = np.array([192, 256])
    results['ann_info']['heatmap_size'] = np.array([48, 64])

    # test flip
    random_flip = TopDownRandomFlip(flip_prob=1.)
    results_flip = random_flip(copy.deepcopy(results))
    assert _check_flip(results['img'], results_flip['img'])

    # test random scale and rotate
    random_scale_rotate = TopDownGetRandomScaleRotation(90, 0.3, 1.0)
    results_scale_rotate = random_scale_rotate(copy.deepcopy(results))
    assert results_scale_rotate['rotation'] <= 180
    assert results_scale_rotate['rotation'] >= -180
    assert (results_scale_rotate['scale'] / results['scale'] <= 1.3).all()
    assert (results_scale_rotate['scale'] / results['scale'] >= 0.7).all()

    # test halfbody transform
    halfbody_transform = TopDownHalfBodyTransform(
        num_joints_half_body=8, prob_half_body=1.)
    results_halfbody = halfbody_transform(copy.deepcopy(results))
    assert (results_halfbody['scale'] <= results['scale']).all()

    affine_transform = TopDownAffine()
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

    generate_target = TopDownGenerateTarget(
        sigma=2, target_type='GaussianHeatMap', unbiased_encoding=True)
    results_target = generate_target(copy.deepcopy(results_tensor))
    assert 'target' in results_target
    assert results_target['target'].shape == (
        num_joints, results['ann_info']['heatmap_size'][1],
        results['ann_info']['heatmap_size'][0])
    assert 'target_weight' in results_target
    assert results_target['target_weight'].shape == (num_joints, 1)

    generate_target = TopDownGenerateTarget(sigma=2, unbiased_encoding=False)
    results_target = generate_target(copy.deepcopy(results_tensor))
    assert 'target' in results_target
    assert results_target['target'].shape == (
        num_joints, results['ann_info']['heatmap_size'][1],
        results['ann_info']['heatmap_size'][0])
    assert 'target_weight' in results_target
    assert results_target['target_weight'].shape == (num_joints, 1)

    generate_target = TopDownGenerateTarget(
        sigma=[2, 3], unbiased_encoding=False)
    results_target = generate_target(copy.deepcopy(results_tensor))
    assert 'target' in results_target
    assert results_target['target'].shape == (
        2, num_joints, results['ann_info']['heatmap_size'][1],
        results['ann_info']['heatmap_size'][0])
    assert 'target_weight' in results_target
    assert results_target['target_weight'].shape == (2, num_joints, 1)

    generate_target = TopDownGenerateTarget(
        sigma=2, encoding='UDP', target_type='GaussianHeatmap')
    results_target = generate_target(copy.deepcopy(results_tensor))
    assert 'target' in results_target
    assert results_target['target'].shape == (
        num_joints, results['ann_info']['heatmap_size'][1],
        results['ann_info']['heatmap_size'][0])
    assert 'target_weight' in results_target
    assert results_target['target_weight'].shape == (num_joints, 1)

    generate_target = TopDownGenerateTarget(
        kernel=(11, 11), encoding='Megvii', unbiased_encoding=False)
    results_target = generate_target(copy.deepcopy(results_tensor))
    assert 'target' in results_target
    assert results_target['target'].shape == (
        num_joints, results['ann_info']['heatmap_size'][1],
        results['ann_info']['heatmap_size'][0])
    assert 'target_weight' in results_target
    assert results_target['target_weight'].shape == (num_joints, 1)

    generate_target = TopDownGenerateTarget(
        kernel=[(11, 11), (7, 7)], encoding='Megvii', unbiased_encoding=False)
    results_target = generate_target(copy.deepcopy(results_tensor))
    assert 'target' in results_target
    assert results_target['target'].shape == (
        2, num_joints, results['ann_info']['heatmap_size'][1],
        results['ann_info']['heatmap_size'][0])
    assert 'target_weight' in results_target
    assert results_target['target_weight'].shape == (2, num_joints, 1)

    collect = Collect(
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ])
    results_final = collect(results_target)
    assert 'img_size' not in results_final['img_metas'].data
    assert 'image_file' in results_final['img_metas'].data


def test_top_down_get_bbox_center_scale():
    # Test conversion from bbox to center and scale
    bbox = np.array([50, 50, 100, 100], dtype=np.float32)
    img_w, img_h = 192, 256
    padding = 1.25

    results = dict(bbox=bbox, ann_info=dict(image_size=[img_w, img_h]))

    pipeline = TopDownGetBboxCenterScale(padding=padding)

    results = pipeline(results)
    center, scale = results['center'], results['scale']
    center_exp = bbox[:2] + bbox[2:] * 0.5
    scale_exp = np.array([bbox[2], bbox[2] / img_w * img_h],
                         dtype=np.float32) / 200 * padding
    np.testing.assert_almost_equal(center, center_exp)
    np.testing.assert_almost_equal(scale, scale_exp)

    # Test using existing center and scale
    center = np.array([100, 100], dtype=np.float32)
    scale = np.array([0.5, 0.5], dtype=np.float32)
    padding = 1.25
    results = dict(center=center.copy(), scale=scale.copy())

    pipeline = TopDownGetBboxCenterScale(padding=padding)

    with warnings.catch_warnings(record=True):
        results = pipeline(results)

    np.testing.assert_almost_equal(scale * padding, results['scale'])


def test_top_down_random_shift_bbox_center_scale():
    center = np.array([100, 100], dtype=np.float32)
    scale = np.array([0.5, 0.5], dtype=np.float32)
    shift_factor = 0.16
    pixel_std = 200.
    results = dict(center=center.copy(), scale=scale.copy())

    pipeline = TopDownRandomShiftBboxCenter(shift_factor=0.16, prob=1.0)
    results = pipeline(results)

    np.testing.assert_array_less(
        np.abs(center - results['center']), scale * shift_factor * pixel_std)
