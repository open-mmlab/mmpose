# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmpose.datasets.builder import PIPELINES
from .bottom_up_transform import HeatmapGenerator


class OffsetGenerator:
    """Generate offset maps for one-stage models."""

    def __init__(self, output_size, num_joints, radius=4):
        if not isinstance(output_size, np.ndarray):
            output_size = np.array(output_size)
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size],
                                        dtype=np.int)
        self.num_joints = num_joints
        if radius < 0:
            radius = self.output_size.prod()**0.5 / 32
        self.radius = radius

    def __call__(self, center, joints, area):
        """Generate offset maps."""

        offset_map = np.zeros(
            (self.num_joints * 2, self.output_size[1], self.output_size[0]),
            dtype=np.float32)
        weight_map = np.zeros(
            (self.num_joints * 2, self.output_size[1], self.output_size[0]),
            dtype=np.float32)
        area_map = np.zeros((self.output_size[1], self.output_size[0]),
                            dtype=np.float32)

        for i in range(len(center)):
            x_center, y_center = center[i, 0, 0], center[i, 0, 1]
            if center[i, 0, 2] < 1 or x_center < 0 or y_center < 0 \
                    or x_center >= self.output_size[0] \
                    or y_center >= self.output_size[1]:
                continue

            for j in range(self.num_joints):
                x, y = joints[i, j, :2]
                if joints[i, j, 2] < 1 or x >= self.output_size[0] \
                        or y >= self.output_size[1] or x < 0 or y < 0:
                    continue

                start_x = max(int(x_center - self.radius), 0)
                start_y = max(int(y_center - self.radius), 0)
                end_x = min(int(x_center + self.radius), self.output_size[0])
                end_y = min(int(y_center + self.radius), self.output_size[1])

                for pos_x in range(start_x, end_x):
                    for pos_y in range(start_y, end_y):
                        offset_x = pos_x - x
                        offset_y = pos_y - y
                        if offset_map[j*2, pos_y, pos_x] != 0 \
                                or offset_map[j*2+1, pos_y, pos_x] != 0:
                            if area_map[pos_y, pos_x] < area[i]:
                                continue
                        offset_map[j * 2, pos_y, pos_x] = offset_x
                        offset_map[j * 2 + 1, pos_y, pos_x] = offset_y
                        weight_map[j * 2, pos_y, pos_x] = 1. / np.sqrt(area[i])
                        weight_map[j * 2 + 1, pos_y,
                                   pos_x] = 1. / np.sqrt(area[i])
                        area_map[pos_y, pos_x] = area[i]

        return offset_map, weight_map


@PIPELINES.register_module()
class OneStageGeneratePersonCenter:

    def __call__(self, results):
        """Copmute center from visible keypoints for each person.

        Note:
            - num_people: P
            - num_keypoints: K

        Args:
            joints (np.ndarray[P, K, 3]): keypoints coordinates and visibility

        Returns:
            center (np.ndarray[P, 1, 3]): person center
        """

        center_list = []
        area_list = []

        for joints in results['joints']:

            area = np.zeros((joints.shape[0]), dtype=np.float32)
            center = np.zeros((joints.shape[0], 1, 3), dtype=np.float32)
            for i in range(joints.shape[0]):
                visible_joints = joints[i][joints[i][..., 2] > 0][..., :2]
                if visible_joints.size == 0:
                    continue

                center[i, 0, :2] = visible_joints.mean(axis=0, keepdims=True)
                center[i, 0, 2] = 1

                area[i] = np.power(
                    visible_joints.max(axis=0) - visible_joints.min(axis=0),
                    2)[:2].sum()

                if area[i] < 64:
                    center[i, 0, 2] = 0

            center_list.append(center)
            area_list.append(area)

        results['center'] = center_list
        results['area'] = area_list

        return results


@PIPELINES.register_module()
class OneStageGenerateHeatmapTarget:
    """Generate multi-scale heatmap target for one-stage.

    Args:
        sigma (int or tuple): Sigma of heatmap Gaussian for center and
            keypoints.
        use_keypoint (bool): Whether to generage heatmap target for keypoints.
            Default: False
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, sigma, use_keypoint=False, use_udp=False):
        if isinstance(sigma, int):
            sigma = (sigma, )
        if use_keypoint:
            assert len(sigma) == 2, 'sigma for keypoints mush be given '\
                                    'if `use_keypoint` is True'
        self.sigma = sigma
        self.use_keypoint = use_keypoint
        self.use_udp = use_udp

    def _generate(self, num_joints, sigma, heatmap_size):
        """Get heatmap generator."""
        heatmap_generator = [
            HeatmapGenerator(output_size, num_joints, sigma, self.use_udp)
            for output_size in heatmap_size
        ]
        return heatmap_generator

    def __call__(self, results):
        """Generate multi-scale heatmap target for one-stage."""
        target_list = list()
        joints_list = results['joints']
        center_list = results['center']
        mask_list = results['mask']

        center_heatmap_generator = self._generate(
            1, self.sigma[0], results['ann_info']['heatmap_size'])

        for scale_id in range(results['ann_info']['num_scales']):
            heatmaps = center_heatmap_generator[scale_id](
                center_list[scale_id])
            target_list.append(heatmaps.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)

        if self.use_keypoint:
            keypoint_heatmap_generator = \
                self._generate(results['ann_info']['num_joints'],
                               self.sigma[1],
                               results['ann_info']['heatmap_size'])

            for scale_id in range(results['ann_info']['num_scales']):
                heatmaps = keypoint_heatmap_generator[scale_id](
                    joints_list[scale_id])
                heatmaps = heatmaps.astype(np.float32)
                target_list[scale_id] = np.concatenate(
                    (target_list[scale_id], heatmaps), axis=0)

        results['heatmaps'] = target_list
        results['masks'] = mask_list

        return results


@PIPELINES.register_module()
class OneStageUpGenerateOffsetTarget:
    """Generate multi-scale offset target for one-stage.

    Args:
        radius (int): Radius of labeled area for each person center.
    """

    def __init__(self, radius=4):
        self.radius = radius

    def _generate(self, num_joints, heatmap_size):
        """Get offset generator."""
        offset_generator = [
            OffsetGenerator(output_size, num_joints, self.radius)
            for output_size in heatmap_size
        ]
        return offset_generator

    def __call__(self, results):
        """Generate multi-scale heatmap target for one-stage."""
        target_list = list()
        weight_list = list()
        center_list = results['center']
        joints_list = results['joints']
        area_list = results['area']

        offset_generator = self._generate(results['ann_info']['num_joints'],
                                          results['ann_info']['heatmap_size'])

        for scale_id in range(results['ann_info']['num_scales']):
            offset, offset_weight = offset_generator[scale_id](
                center_list[scale_id], joints_list[scale_id],
                area_list[scale_id])
            target_list.append(offset.astype(np.float32))
            weight_list.append(offset_weight)
        results['offsets'] = target_list
        results['offset_weights'] = weight_list

        return results
