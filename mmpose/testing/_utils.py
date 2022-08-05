# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.data import InstanceData, PixelData

from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xyxy2cs


def get_packed_inputs(batch_size=2,
                      num_instances=1,
                      num_keypoints=17,
                      num_levels=1,
                      image_shape=(3, 128, 128),
                      input_size=(192, 256),
                      heatmap_size=(48, 64),
                      simcc_split_ratio=2.0,
                      with_heatmap=True,
                      with_reg_label=True,
                      with_simcc_label=None):
    """Create a dummy batch of model inputs and data samples."""
    rng = np.random.RandomState(0)

    packed_inputs = []
    for idx in range(batch_size):
        inputs = dict()

        # input
        c, h, w = image_shape
        image = rng.randint(0, 255, size=(c, h, w), dtype=np.uint8)
        inputs['inputs'] = torch.from_numpy(image)

        # meta
        img_meta = {
            'id': idx,
            'img_id': idx,
            'img_path': '<demo>.png',
            'img_shape': image_shape,
            'input_size': input_size,
            'flip': False,
            'flip_direction': None
        }

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
            if with_simcc_label == 'gaussian':
                len_x = np.around(input_size[0] * simcc_split_ratio)
                len_y = np.around(input_size[1] * simcc_split_ratio)
                gt_instance_labels.keypoint_x_labels = torch.FloatTensor(
                    _rand_simcc_label(rng, num_instances, num_keypoints,
                                      len_x))
                gt_instance_labels.keypoint_y_labels = torch.FloatTensor(
                    _rand_simcc_label(rng, num_instances, num_keypoints,
                                      len_y))
            else:
                gt_instance_labels.keypoint_x_labels = torch.LongTensor(
                    _rand_keypoints(rng, bboxes, num_keypoints)[..., 0])
                gt_instance_labels.keypoint_y_labels = torch.LongTensor(
                    _rand_keypoints(rng, bboxes, num_keypoints)[..., 1])

        # gt_fields
        gt_fields = PixelData()
        if with_heatmap:
            if num_levels == 1:
                # generate single-scale heatmaps
                W, H = heatmap_size
                heatmaps = rng.rand(num_keypoints, H, W)
                gt_fields.heatmaps = torch.FloatTensor(heatmaps)
            else:
                # generate multi-scale heatmaps
                heatmaps = []
                for _ in range(num_levels):
                    W, H = heatmap_size
                    heatmaps_ = rng.rand(num_keypoints, H, W)
                    heatmaps.append(heatmaps_)
                # [num_levels*K, H, W]
                heatmaps = np.concatenate(heatmaps)
                gt_fields.heatmaps = torch.FloatTensor(heatmaps)

        data_sample.gt_instances = gt_instances
        data_sample.gt_instance_labels = gt_instance_labels
        data_sample.gt_fields = gt_fields

        inputs['data_sample'] = data_sample
        packed_inputs.append(inputs)

    return packed_inputs


def _rand_keypoints(rng, bboxes, num_keypoints):
    n = bboxes.shape[0]
    keypoints = rng.rand(n, num_keypoints,
                         2) * bboxes[:, None, 2:4] + bboxes[:, None, :2]
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
