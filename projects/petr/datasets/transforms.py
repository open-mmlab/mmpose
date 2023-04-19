# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox.box_type import autocast_box_type
from mmengine.structures import PixelData

from mmpose.codecs.utils import generate_gaussian_heatmaps
from .bbox_keypoint_structure import BBoxKeypoints


@TRANSFORMS.register_module(force=True)
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


@TRANSFORMS.register_module(force=True)
class PackDetPoseInputs(PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_keypoints': 'keypoints',
        'gt_keypoints_visible': 'keypoints_visible'
    }
    field_mapping_table = {
        'gt_heatmaps': 'gt_heatmaps',
    }

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'ori_shape',
                            'img_shape', 'scale_factor', 'flip',
                            'flip_direction', 'flip_indices', 'raw_ann_info'),
                 pack_transformed=False):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        results['gt_keypoints'] = results['gt_bboxes'].keypoints
        results['gt_keypoints_visible'] = results[
            'gt_bboxes'].keypoints_visible

        # pack fields
        gt_fields = None
        for key, packed_key in self.field_mapping_table.items():
            if key in results:

                if gt_fields is None:
                    gt_fields = PixelData()
                else:
                    assert isinstance(
                        gt_fields, PixelData
                    ), 'Got mixed single-level and multi-level pixel data.'

                gt_fields.set_field(results[key], packed_key)

        # Ensure all keys in `self.meta_keys` are in the `results` dictionary,
        # which is necessary for `PackDetInputs` but not guaranteed during
        # inference with an inferencer
        for key in self.meta_keys:
            if key not in results:
                results[key] = None

        results = super().transform(results)
        if gt_fields:
            results['data_samples'].gt_fields = gt_fields.to_tensor()

        return results


@TRANSFORMS.register_module(force=True)
class GenerateHeatmap(BaseTransform):

    def _get_instance_wise_sigmas(self,
                                  bbox: np.ndarray,
                                  heatmap_min_overlap: float = 0.9
                                  ) -> np.ndarray:
        """Get sigma values for each instance according to their size.

        Args:
            bbox (np.ndarray): Bounding box in shape (N, 4, 2)

        Returns:
            np.ndarray: Array containing the sigma values for each instance.
        """
        sigmas = np.zeros((bbox.shape[0], ), dtype=np.float32)

        heights = bbox[:, 3] - bbox[:, 1]
        widths = bbox[:, 2] - bbox[:, 0]

        for i in range(bbox.shape[0]):
            h, w = heights[i], widths[i]

            # compute sigma for each instance
            # condition 1
            a1, b1 = 1, h + w
            c1 = w * h * (1 - heatmap_min_overlap) / (1 + heatmap_min_overlap)
            sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
            r1 = (b1 + sq1) / 2

            # condition 2
            a2 = 4
            b2 = 2 * (h + w)
            c2 = (1 - heatmap_min_overlap) * w * h
            sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
            r2 = (b2 + sq2) / 2

            # condition 3
            a3 = 4 * heatmap_min_overlap
            b3 = -2 * heatmap_min_overlap * (h + w)
            c3 = (heatmap_min_overlap - 1) * w * h
            sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
            r3 = (b3 + sq3) / 2

            sigmas[i] = min(r1, r2, r3, 3) / 3

        return sigmas

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'gt_bboxes' in results

        bbox = results['gt_bboxes'].tensor.numpy() / 8
        keypoints = results['gt_bboxes'].keypoints.numpy() / 8
        keypoints_visible = results['gt_bboxes'].keypoints_visible.numpy()

        heatmap_size = [
            results['img_shape'][1] // 8 + 1, results['img_shape'][0] // 8 + 1
        ]
        sigmas = self._get_instance_wise_sigmas(bbox)

        hm, _ = generate_gaussian_heatmaps(heatmap_size, keypoints,
                                           keypoints_visible, sigmas)

        results['gt_heatmaps'] = hm

        return results
