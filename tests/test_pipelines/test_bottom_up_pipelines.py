# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest
import xtcocotools
from xtcocotools.coco import COCO

from mmpose.datasets.pipelines import (BottomUpGenerateHeatmapTarget,
                                       BottomUpGeneratePAFTarget,
                                       BottomUpGenerateTarget,
                                       BottomUpGetImgSize,
                                       BottomUpRandomAffine,
                                       BottomUpRandomFlip, BottomUpResizeAlign,
                                       LoadImageFromFile)


def _get_mask(coco, anno, img_id):
    img_info = coco.loadImgs(img_id)[0]

    m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)

    for obj in anno:
        if obj['iscrowd']:
            rle = xtcocotools.mask.frPyObjects(obj['segmentation'],
                                               img_info['height'],
                                               img_info['width'])
            m += xtcocotools.mask.decode(rle)
        elif obj['num_keypoints'] == 0:
            rles = xtcocotools.mask.frPyObjects(obj['segmentation'],
                                                img_info['height'],
                                                img_info['width'])
            for rle in rles:
                m += xtcocotools.mask.decode(rle)

    return m < 0.5


def _get_joints(anno, ann_info, int_sigma):
    num_people = len(anno)

    if ann_info['scale_aware_sigma']:
        joints = np.zeros((num_people, ann_info['num_joints'], 4),
                          dtype=np.float32)
    else:
        joints = np.zeros((num_people, ann_info['num_joints'], 3),
                          dtype=np.float32)

    for i, obj in enumerate(anno):
        joints[i, :ann_info['num_joints'], :3] = \
            np.array(obj['keypoints']).reshape([-1, 3])
        if ann_info['scale_aware_sigma']:
            # get person box
            box = obj['bbox']
            size = max(box[2], box[3])
            sigma = size / 256 * 2
            if int_sigma:
                sigma = int(np.ceil(sigma))
            assert sigma > 0, sigma
            joints[i, :, 3] = sigma

    return joints


def _check_flip(origin_imgs, result_imgs):
    """Check if the origin_imgs are flipped correctly."""
    h, w, c = origin_imgs.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                if result_imgs[i, j, k] != origin_imgs[i, w - 1 - j, k]:
                    return False
    return True


def test_bottomup_pipeline():

    data_prefix = 'tests/data/coco/'
    ann_file = osp.join(data_prefix, 'test_coco.json')
    coco = COCO(ann_file)

    ann_info = {}
    ann_info['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                              [11, 12], [13, 14], [15, 16]]
    ann_info['flip_index'] = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ]

    ann_info['use_different_joint_weights'] = False
    ann_info['joint_weights'] = np.array([
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
                                         dtype=np.float32).reshape((17, 1))
    ann_info['image_size'] = np.array([384, 512])
    ann_info['heatmap_size'] = np.array([[96, 128], [192, 256]])
    ann_info['num_joints'] = 17
    ann_info['num_scales'] = 2
    ann_info['scale_aware_sigma'] = False

    ann_ids = coco.getAnnIds(785)
    anno = coco.loadAnns(ann_ids)
    mask = _get_mask(coco, anno, 785)

    anno = [
        obj for obj in anno if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
    ]
    joints = _get_joints(anno, ann_info, False)

    mask_list = [mask.copy() for _ in range(ann_info['num_scales'])]
    joints_list = [joints.copy() for _ in range(ann_info['num_scales'])]

    results = {}
    results['dataset'] = 'coco'
    results['image_file'] = osp.join(data_prefix, '000000000785.jpg')
    results['mask'] = mask_list
    results['joints'] = joints_list
    results['ann_info'] = ann_info

    transform = LoadImageFromFile()
    results = transform(copy.deepcopy(results))
    assert results['img'].shape == (425, 640, 3)

    # test HorizontalFlip
    random_horizontal_flip = BottomUpRandomFlip(flip_prob=1.)
    results_horizontal_flip = random_horizontal_flip(copy.deepcopy(results))
    assert _check_flip(results['img'], results_horizontal_flip['img'])

    random_horizontal_flip = BottomUpRandomFlip(flip_prob=0.)
    results_horizontal_flip = random_horizontal_flip(copy.deepcopy(results))
    assert (results['img'] == results_horizontal_flip['img']).all()

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    # test TopDownAffine
    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'short', 0)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (512, 384, 3)

    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'short',
                                                   40)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (512, 384, 3)

    results_copy = copy.deepcopy(results)
    results_copy['ann_info']['scale_aware_sigma'] = True
    joints = _get_joints(anno, results_copy['ann_info'], False)
    results_copy['joints'] = \
        [joints.copy() for _ in range(results_copy['ann_info']['num_scales'])]
    results_affine_transform = random_affine_transform(results_copy)
    assert results_affine_transform['img'].shape == (512, 384, 3)

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'long', 40)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (512, 384, 3)

    with pytest.raises(ValueError):
        random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5],
                                                       'short-long', 40)
        results_affine_transform = random_affine_transform(
            copy.deepcopy(results))

    # test BottomUpGenerateTarget
    generate_multi_target = BottomUpGenerateTarget(2, 30)
    results_generate_multi_target = generate_multi_target(
        copy.deepcopy(results))
    assert 'targets' in results_generate_multi_target
    assert len(results_generate_multi_target['targets']
               ) == results['ann_info']['num_scales']

    # test BottomUpGetImgSize when W > H
    get_multi_scale_size = BottomUpGetImgSize([1])
    results_get_multi_scale_size = get_multi_scale_size(copy.deepcopy(results))
    assert 'test_scale_factor' in results_get_multi_scale_size['ann_info']
    assert 'base_size' in results_get_multi_scale_size['ann_info']
    assert 'center' in results_get_multi_scale_size['ann_info']
    assert 'scale' in results_get_multi_scale_size['ann_info']
    assert results_get_multi_scale_size['ann_info']['base_size'][1] == 512

    # test BottomUpResizeAlign
    transforms = [
        dict(type='ToTensor'),
        dict(
            type='NormalizeTensor',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ]
    resize_align_multi_scale = BottomUpResizeAlign(transforms=transforms)
    results_copy = copy.deepcopy(results_get_multi_scale_size)
    results_resize_align_multi_scale = resize_align_multi_scale(results_copy)
    assert 'aug_data' in results_resize_align_multi_scale['ann_info']

    # test when W < H
    ann_info['image_size'] = np.array([512, 384])
    ann_info['heatmap_size'] = np.array([[128, 96], [256, 192]])
    results = {}
    results['dataset'] = 'coco'
    results['image_file'] = osp.join(data_prefix, '000000000785.jpg')
    results['mask'] = mask_list
    results['joints'] = joints_list
    results['ann_info'] = ann_info
    results['img'] = np.random.rand(640, 425, 3)

    # test HorizontalFlip
    random_horizontal_flip = BottomUpRandomFlip(flip_prob=1.)
    results_horizontal_flip = random_horizontal_flip(copy.deepcopy(results))
    assert _check_flip(results['img'], results_horizontal_flip['img'])

    random_horizontal_flip = BottomUpRandomFlip(flip_prob=0.)
    results_horizontal_flip = random_horizontal_flip(copy.deepcopy(results))
    assert (results['img'] == results_horizontal_flip['img']).all()

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_horizontal_flip(
            copy.deepcopy(results_copy))

    # test TopDownAffine
    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'short', 0)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (384, 512, 3)

    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'short',
                                                   40)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (384, 512, 3)

    results_copy = copy.deepcopy(results)
    results_copy['ann_info']['scale_aware_sigma'] = True
    joints = _get_joints(anno, results_copy['ann_info'], False)
    results_copy['joints'] = \
        [joints.copy() for _ in range(results_copy['ann_info']['num_scales'])]
    results_affine_transform = random_affine_transform(results_copy)
    assert results_affine_transform['img'].shape == (384, 512, 3)

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[0]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['joints'] = joints_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    results_copy = copy.deepcopy(results)
    results_copy['mask'] = mask_list[:1]
    with pytest.raises(AssertionError):
        results_horizontal_flip = random_affine_transform(
            copy.deepcopy(results_copy))

    random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5], 'long', 40)
    results_affine_transform = random_affine_transform(copy.deepcopy(results))
    assert results_affine_transform['img'].shape == (384, 512, 3)

    with pytest.raises(ValueError):
        random_affine_transform = BottomUpRandomAffine(30, [0.75, 1.5],
                                                       'short-long', 40)
        results_affine_transform = random_affine_transform(
            copy.deepcopy(results))

    # test BottomUpGenerateTarget
    generate_multi_target = BottomUpGenerateTarget(2, 30)
    results_generate_multi_target = generate_multi_target(
        copy.deepcopy(results))
    assert 'targets' in results_generate_multi_target
    assert len(results_generate_multi_target['targets']
               ) == results['ann_info']['num_scales']

    # test BottomUpGetImgSize when W < H
    get_multi_scale_size = BottomUpGetImgSize([1])
    results_get_multi_scale_size = get_multi_scale_size(copy.deepcopy(results))
    assert 'test_scale_factor' in results_get_multi_scale_size['ann_info']
    assert 'base_size' in results_get_multi_scale_size['ann_info']
    assert 'center' in results_get_multi_scale_size['ann_info']
    assert 'scale' in results_get_multi_scale_size['ann_info']
    assert results_get_multi_scale_size['ann_info']['base_size'][0] == 512


def test_BottomUpGenerateHeatmapTarget():

    data_prefix = 'tests/data/coco/'
    ann_file = osp.join(data_prefix, 'test_coco.json')
    coco = COCO(ann_file)

    ann_info = {}
    ann_info['heatmap_size'] = np.array([128, 256])
    ann_info['num_joints'] = 17
    ann_info['num_scales'] = 2
    ann_info['scale_aware_sigma'] = False

    ann_ids = coco.getAnnIds(785)
    anno = coco.loadAnns(ann_ids)
    mask = _get_mask(coco, anno, 785)

    anno = [
        obj for obj in anno if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
    ]
    joints = _get_joints(anno, ann_info, False)

    mask_list = [mask.copy() for _ in range(ann_info['num_scales'])]
    joints_list = [joints.copy() for _ in range(ann_info['num_scales'])]

    results = {}
    results['dataset'] = 'coco'
    results['image_file'] = osp.join(data_prefix, '000000000785.jpg')
    results['mask'] = mask_list
    results['joints'] = joints_list
    results['ann_info'] = ann_info

    generate_heatmap_target = BottomUpGenerateHeatmapTarget(2)
    results_generate_heatmap_target = generate_heatmap_target(results)
    assert 'target' in results_generate_heatmap_target
    assert len(results_generate_heatmap_target['target']
               ) == results['ann_info']['num_scales']


def test_BottomUpGeneratePAFTarget():

    ann_info = {}
    ann_info['skeleton'] = [[0, 1], [2, 3]]
    ann_info['heatmap_size'] = np.array([5])
    ann_info['num_joints'] = 4
    ann_info['num_scales'] = 1

    mask = np.ones((5, 5), dtype=bool)
    joints = np.array([[[1, 1, 2], [3, 3, 2], [0, 0, 0], [0, 0, 0]],
                       [[1, 3, 2], [3, 1, 2], [0, 0, 0], [0, 0, 0]]])

    mask_list = [mask.copy() for _ in range(ann_info['num_scales'])]
    joints_list = [joints.copy() for _ in range(ann_info['num_scales'])]

    results = {}
    results['dataset'] = 'coco'
    results['mask'] = mask_list
    results['joints'] = joints_list
    results['ann_info'] = ann_info

    generate_paf_target = BottomUpGeneratePAFTarget(1)
    results_generate_paf_target = generate_paf_target(results)
    sqrt = np.sqrt(2) / 2
    assert (results_generate_paf_target['target'] == np.array(
        [[[sqrt, sqrt, 0, sqrt, sqrt], [sqrt, sqrt, sqrt, sqrt, sqrt],
          [0, sqrt, sqrt, sqrt, 0], [sqrt, sqrt, sqrt, sqrt, sqrt],
          [sqrt, sqrt, 0, sqrt, sqrt]],
         [[sqrt, sqrt, 0, -sqrt, -sqrt], [sqrt, sqrt, 0, -sqrt, -sqrt],
          [0, 0, 0, 0, 0], [-sqrt, -sqrt, 0, sqrt, sqrt],
          [-sqrt, -sqrt, 0, sqrt, sqrt]],
         [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]],
        dtype=np.float32)).all()
