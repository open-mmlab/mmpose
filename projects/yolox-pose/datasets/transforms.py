# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import FilterAnnotations as FilterDetAnnotations
from mmdet.datasets.transforms import PackDetInputs
from mmdet.structures.bbox.box_type import autocast_box_type
from mmyolo.registry import TRANSFORMS

from .bbox_keypoint_structure import BBoxKeypoints


@TRANSFORMS.register_module()
class PoseToDetConverter(BaseTransform):
    """This transform converts the pose data element into a format that is
    suitable for the mmdet transforms."""

    def transform(self, results: dict) -> dict:

        results['seg_map_path'] = None
        results['height'] = results['img_shape'][0]
        results['width'] = results['img_shape'][1]

        num_instances = len(results.get('bbox', []))

        if num_instances == 0:
            results['bbox'] = np.empty((0, 4), dtype=np.float32)
            results['keypoints'] = np.empty(
                (0, len(results['flip_indices']), 2), dtype=np.float32)
            results['keypoints_visible'] = np.empty(
                (0, len(results['flip_indices'])), dtype=np.int32)
            results['category_id'] = []

        results['gt_bboxes'] = BBoxKeypoints(
            data=results['bbox'],
            keypoints=results['keypoints'],
            keypoints_visible=results['keypoints_visible'],
            flip_indices=results['flip_indices'],
        )

        results['gt_ignore_flags'] = np.array([False] * num_instances)
        results['gt_bboxes_labels'] = np.array(results['category_id']) - 1

        return results


@TRANSFORMS.register_module()
class PackDetPoseInputs(PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_keypoints': 'keypoints',
        'gt_keypoints_visible': 'keypoints_visible'
    }

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'ori_shape',
                            'img_shape', 'scale_factor', 'flip',
                            'flip_direction', 'flip_indices', 'raw_ann_info'),
                 pack_transformed=False):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        # Add keypoints and their visibility to the results dictionary
        results['gt_keypoints'] = results['gt_bboxes'].keypoints
        results['gt_keypoints_visible'] = results[
            'gt_bboxes'].keypoints_visible

        # Ensure all keys in `self.meta_keys` are in the `results` dictionary,
        # which is necessary for `PackDetInputs` but not guaranteed during
        # inference with an inferencer
        for key in self.meta_keys:
            if key not in results:
                results[key] = None
        return super().transform(results)


@TRANSFORMS.register_module()
class FilterDetPoseAnnotations(FilterDetAnnotations):
    """Filter invalid annotations.

    In addition to the conditions checked by ``FilterDetAnnotations``, this
    filter adds a new condition requiring instances to have at least one
    visible keypoints.
    """

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return results

        tests = []
        if self.by_box:
            tests.append(((gt_bboxes.widths > self.min_gt_bbox_wh[0]) &
                          (gt_bboxes.heights > self.min_gt_bbox_wh[1]) &
                          (gt_bboxes.num_keypoints > 0)).numpy())

        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep.any():
            if self.keep_empty:
                return None

        keys = ('gt_bboxes', 'gt_bboxes_labels', 'gt_masks', 'gt_ignore_flags')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]

        return results
