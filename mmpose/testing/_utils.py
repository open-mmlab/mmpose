# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.data import InstanceData, PixelData

from mmpose.core.bbox.transforms import bbox_xywh2cs, bbox_xyxy2xywh
from mmpose.core.data_structures import PoseDataSample


def get_packed_inputs(batch_size=2,
                      num_instances=1,
                      num_keypoints=17,
                      image_shape=(3, 128, 128),
                      input_size=(192, 256),
                      heatmap_size=(48, 64),
                      with_heatmap=True,
                      with_reg_label=True):
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

        bboxes_xyxy = _rand_bboxes(rng, num_instances, w, h)
        bboxes = bbox_xyxy2xywh(bboxes_xyxy)
        bbox_centers, bbox_scales = bbox_xywh2cs(bboxes)
        gt_instances.bboxes = torch.FloatTensor(bboxes)
        gt_instances.bbox_centers = torch.FloatTensor(bbox_centers)
        gt_instances.bbox_scales = torch.FloatTensor(bbox_scales)

        keypoints = _rand_keypoints(rng, bboxes, num_keypoints)
        keypoints_visible = np.ones((num_instances, num_keypoints),
                                    dtype=np.float32)
        keypoint_weights = keypoints_visible.copy()
        gt_instances.keypoints = torch.FloatTensor(keypoints)
        gt_instances.keypoints_visible = torch.FloatTensor(keypoints_visible)
        gt_instances.keypoint_weights = torch.FloatTensor(keypoint_weights)

        if with_reg_label:
            gt_instances.reg_label = torch.FloatTensor(keypoints / input_size)

        data_sample.gt_instances = gt_instances

        # gt_fields
        gt_fields = PixelData()
        if with_heatmap:
            W, H = heatmap_size
            heatmaps = rng.rand(num_keypoints, H, W)
            gt_fields.heatmaps = torch.FloatTensor(heatmaps)

        data_sample.gt_fields = gt_fields

        inputs['data_sample'] = data_sample
        packed_inputs.append(inputs)

    return packed_inputs


def _rand_keypoints(rng, bboxes, num_keypoints):
    n = bboxes.shape[0]
    keypoints = rng.rand(n, num_keypoints,
                         2) * bboxes[:, None, 2:4] + bboxes[:, None, :2]
    return keypoints


def _rand_bboxes(rng, num_instances, w, h):
    cx, cy, bw, bh = rng.rand(num_instances, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clip(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clip(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clip(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clip(0, h)

    bboxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
    return bboxes
