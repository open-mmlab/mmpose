# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_list_of

from .bbox.transforms import get_warp_matrix
from .pose_data_sample import PoseDataSample


def merge_data_samples(data_samples: List[PoseDataSample]) -> PoseDataSample:
    """Merge the given data samples into a single data sample.

    This function can be used to merge the top-down predictions with
    bboxes from the same image. The merged data sample will contain all
    instances from the input data samples, and the identical metainfo with
    the first input data sample.

    Args:
        data_samples (List[:obj:`PoseDataSample`]): The data samples to
            merge

    Returns:
        PoseDataSample: The merged data sample.
    """

    if not is_list_of(data_samples, PoseDataSample):
        raise ValueError('Invalid input type, should be a list of '
                         ':obj:`PoseDataSample`')

    assert len(data_samples) > 0

    merged = PoseDataSample(metainfo=data_samples[0].metainfo)

    if 'gt_instances' in data_samples[0]:
        merged.gt_instances = InstanceData.cat(
            [d.gt_instances for d in data_samples])

    if 'pred_instances' in data_samples[0]:
        merged.pred_instances = InstanceData.cat(
            [d.pred_instances for d in data_samples])

    if 'pred_fields' in data_samples[0] and 'heatmaps' in data_samples[
            0].pred_fields:
        reverted_heatmaps = [
            revert_heatmap(data_sample.pred_fields.heatmaps,
                           data_sample.gt_instances.bbox_centers,
                           data_sample.gt_instances.bbox_scales,
                           data_sample.ori_shape)
            for data_sample in data_samples
        ]

        merged_heatmaps = np.max(reverted_heatmaps, axis=0)
        pred_fields = PixelData()
        pred_fields.set_data(dict(heatmaps=merged_heatmaps))
        merged.pred_fields = pred_fields

    if 'gt_fields' in data_samples[0] and 'heatmaps' in data_samples[
            0].gt_fields:
        reverted_heatmaps = [
            revert_heatmap(data_sample.gt_fields.heatmaps,
                           data_sample.gt_instances.bbox_centers,
                           data_sample.gt_instances.bbox_scales,
                           data_sample.ori_shape)
            for data_sample in data_samples
        ]

        merged_heatmaps = np.max(reverted_heatmaps, axis=0)
        gt_fields = PixelData()
        gt_fields.set_data(dict(heatmaps=merged_heatmaps))
        merged.gt_fields = gt_fields

    return merged


def revert_heatmap(heatmap, bbox_center, bbox_scale, img_shape):
    """Revert predicted heatmap on the original image.

    Args:
        heatmap (np.ndarray or torch.tensor): predicted heatmap.
        bbox_center (np.ndarray): bounding box center coordinate.
        bbox_scale (np.ndarray): bounding box scale.
        img_shape (tuple or list): size of original image.
    """
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().detach().numpy()

    ndim = heatmap.ndim
    # [K, H, W] -> [H, W, K]
    if ndim == 3:
        heatmap = heatmap.transpose(1, 2, 0)

    hm_h, hm_w = heatmap.shape[:2]
    img_h, img_w = img_shape
    warp_mat = get_warp_matrix(
        bbox_center.reshape((2, )),
        bbox_scale.reshape((2, )),
        rot=0,
        output_size=(hm_w, hm_h),
        inv=True)

    heatmap = cv2.warpAffine(
        heatmap, warp_mat, (img_w, img_h), flags=cv2.INTER_LINEAR)

    # [H, W, K] -> [K, H, W]
    if ndim == 3:
        heatmap = heatmap.transpose(2, 0, 1)

    return heatmap
