# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData

from mmpose.structures import MultilevelPixelData, PoseDataSample
from mmpose.structures.bbox import bbox_xyxy2cs


def get_coco_sample(
        img_shape=(240, 320),
        img_fill: Optional[int] = None,
        num_instances=1,
        with_bbox_cs=True,
        with_img_mask=False,
        random_keypoints_visible=False,
        non_occlusion=False):
    """Create a dummy data sample in COCO style."""
    rng = np.random.RandomState(0)
    h, w = img_shape
    if img_fill is None:
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        img = np.full((h, w, 3), img_fill, dtype=np.uint8)

    if non_occlusion:
        bbox = _rand_bboxes(rng, num_instances, w / num_instances, h)
        for i in range(num_instances):
            bbox[i, 0::2] += w / num_instances * i
    else:
        bbox = _rand_bboxes(rng, num_instances, w, h)

    keypoints = _rand_keypoints(rng, bbox, 17)
    if random_keypoints_visible:
        keypoints_visible = np.random.randint(0, 2, (num_instances,
                                                     17)).astype(np.float32)
    else:
        keypoints_visible = np.full((num_instances, 17), 1, dtype=np.float32)

    upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lower_body_ids = [11, 12, 13, 14, 15, 16]
    flip_pairs = [[2, 1], [1, 2], [4, 3], [3, 4], [6, 5], [5, 6], [8, 7],
                  [7, 8], [10, 9], [9, 10], [12, 11], [11, 12], [14, 13],
                  [13, 14], [16, 15], [15, 16]]
    flip_indices = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    dataset_keypoint_weights = np.array([
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ]).astype(np.float32)

    data = {
        'img': img,
        'img_shape': img_shape,
        'bbox': bbox,
        'keypoints': keypoints,
        'keypoints_visible': keypoints_visible,
        'upper_body_ids': upper_body_ids,
        'lower_body_ids': lower_body_ids,
        'flip_pairs': flip_pairs,
        'flip_indices': flip_indices,
        'dataset_keypoint_weights': dataset_keypoint_weights,
        'invalid_segs': [],
    }

    if with_bbox_cs:
        data['bbox_center'], data['bbox_scale'] = bbox_xyxy2cs(data['bbox'])

    if with_img_mask:
        data['img_mask'] = np.random.randint(0, 2, (h, w), dtype=np.uint8)

    return data


def get_packed_inputs(batch_size=2,
                      num_instances=1,
                      num_keypoints=17,
                      num_levels=1,
                      img_shape=(128, 128),
                      input_size=(192, 256),
                      heatmap_size=(48, 64),
                      simcc_split_ratio=2.0,
                      with_heatmap=True,
                      with_reg_label=True,
                      with_simcc_label=True):
    """Create a dummy batch of model inputs and data samples."""
    rng = np.random.RandomState(0)

    packed_inputs = []
    for idx in range(batch_size):
        inputs = dict()

        # input
        h, w = img_shape
        image = rng.randint(0, 255, size=(3, h, w), dtype=np.uint8)
        inputs['inputs'] = torch.from_numpy(image)

        # meta
        img_meta = {
            'id': idx,
            'img_id': idx,
            'img_path': '<demo>.png',
            'img_shape': img_shape,
            'input_size': input_size,
            'flip': False,
            'flip_direction': None,
            'flip_indices': list(range(num_keypoints))
        }

        np.random.shuffle(img_meta['flip_indices'])
        data_sample = PoseDataSample(metainfo=img_meta)

        # gt_instance
        gt_instances = InstanceData()
        gt_instance_labels = InstanceData()

        bboxes = _rand_bboxes(rng, num_instances, w, h)
        bbox_centers, bbox_scales = bbox_xyxy2cs(bboxes)

        keypoints = _rand_keypoints(rng, bboxes, num_keypoints)
        keypoints_visible = np.ones((num_instances, num_keypoints),
                                    dtype=np.float32)

        # [N, K] -> [N, num_levels, K]
        # keep the first dimension as the num_instances
        if num_levels > 1:
            keypoint_weights = np.tile(keypoints_visible[:, None],
                                       (1, num_levels, 1))
        else:
            keypoint_weights = keypoints_visible.copy()

        gt_instances.bboxes = bboxes
        gt_instances.bbox_centers = bbox_centers
        gt_instances.bbox_scales = bbox_scales
        gt_instances.bbox_scores = np.ones((num_instances, ), dtype=np.float32)
        gt_instances.keypoints = keypoints
        gt_instances.keypoints_visible = keypoints_visible

        gt_instance_labels.keypoint_weights = torch.FloatTensor(
            keypoint_weights)

        if with_reg_label:
            gt_instance_labels.keypoint_labels = torch.FloatTensor(keypoints /
                                                                   input_size)

        if with_simcc_label:
            len_x = np.around(input_size[0] * simcc_split_ratio)
            len_y = np.around(input_size[1] * simcc_split_ratio)
            gt_instance_labels.keypoint_x_labels = torch.FloatTensor(
                _rand_simcc_label(rng, num_instances, num_keypoints, len_x))
            gt_instance_labels.keypoint_y_labels = torch.FloatTensor(
                _rand_simcc_label(rng, num_instances, num_keypoints, len_y))

        # gt_fields
        if with_heatmap:
            if num_levels == 1:
                gt_fields = PixelData()
                # generate single-level heatmaps
                W, H = heatmap_size
                heatmaps = rng.rand(num_keypoints, H, W)
                gt_fields.heatmaps = torch.FloatTensor(heatmaps)
            else:
                # generate multilevel heatmaps
                heatmaps = []
                for _ in range(num_levels):
                    W, H = heatmap_size
                    heatmaps_ = rng.rand(num_keypoints, H, W)
                    heatmaps.append(torch.FloatTensor(heatmaps_))
                # [num_levels*K, H, W]
                gt_fields = MultilevelPixelData()
                gt_fields.heatmaps = heatmaps
            data_sample.gt_fields = gt_fields

        data_sample.gt_instances = gt_instances
        data_sample.gt_instance_labels = gt_instance_labels

        inputs['data_sample'] = data_sample
        packed_inputs.append(inputs)

    return packed_inputs


def _rand_keypoints(rng, bboxes, num_keypoints):
    n = bboxes.shape[0]
    relative_pos = rng.rand(n, num_keypoints, 2)
    keypoints = relative_pos * bboxes[:, None, :2] + (
        1 - relative_pos) * bboxes[:, None, 2:4]

    return keypoints


def _rand_simcc_label(rng, num_instances, num_keypoints, len_feats):
    simcc_label = rng.rand(num_instances, num_keypoints, int(len_feats))
    return simcc_label


def _rand_bboxes(rng, num_instances, img_w, img_h):
    cx, cy, bw, bh = rng.rand(num_instances, 4).T

    tl_x = ((cx * img_w) - (img_w * bw / 2)).clip(0, img_w)
    tl_y = ((cy * img_h) - (img_h * bh / 2)).clip(0, img_h)
    br_x = ((cx * img_w) + (img_w * bw / 2)).clip(0, img_w)
    br_y = ((cy * img_h) + (img_h * bh / 2)).clip(0, img_h)

    bboxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
    return bboxes
