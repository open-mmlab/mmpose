# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv.image import imflip
from mmcv.transforms import BaseTransform
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms.utils import cache_randomness
from mmengine import is_list_of
from scipy.stats import truncnorm

from mmpose.core.bbox import (bbox_xywh2cs, flip_bbox, get_udp_warp_matrix,
                              get_warp_matrix)
from mmpose.core.keypoint import flip_keypoints
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TopDownGetBboxCenterScale(BaseTransform):
    """Convert bboxes from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox
        - input_size

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
        pixel_std (float): The normalization factor to calculate `bbox_scale`.
            Defaults to 200.0
    """

    def __init__(self, padding: float = 1.25, pixel_std: float = 200.) -> None:
        super().__init__()

        self.padding = padding
        self.pixel_std = pixel_std

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`TopDownGetBboxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        if 'bbox_center' in results and 'bbox_scale' in results:
            warnings.warn('Use the existing "bbox_center" and "bbox_scale". '
                          'The padding will still be applied.')
            results['bbox_scale'] *= self.padding

        else:
            bbox = results['bbox']
            w, h = results['input_size']
            aspect_ratio = w / h

            center, scale = bbox_xywh2cs(
                bbox,
                aspect_ratio=aspect_ratio,
                padding=self.padding,
                pixel_std=self.pixel_std)

            results['bbox_center'] = center
            results['bbox_scale'] = scale

        return results


@TRANSFORMS.register_module()
class TopDownRandomFlip(MMCV_RandomFlip):
    """Randomly flip the image, bbox and keypoints.

    Required Keys:

        - img
        - image_shape
        - flip_pairs
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)

    Modified Keys:

        - img
        - bbox_center
        - keypoints
        - keypoints_visible

    Added Keys:

        - flip
        - flip_direction

    Args:
        prob (float | list[float]): The flipping probability. If a list is
            given, the argument `direction` should be a list with the same
            length. And each element in `prob` indicates the flipping
            probability of the corresponding one in ``direction``. Defaults
            to 0.5
        direction (str | list[str]): The flipping direction. Options are
            ``'horizontal'``, ``'vertical'`` and ``'diagonal'``. If a list is
            is given, each data sample's flipping direction will be sampled
            from a distribution determined by the argument ``prob``. Defaults
            to ``'horizontal'``.
    """

    def __init__(self,
                 prob: Union[float, List[float]] = 0.5,
                 direction: Union[str, List[str]] = 'horizontal') -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but \
                              got `{type(prob)}`.')
        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str, \
                               but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      List) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`RandomFlip`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        flip_dir = self._choose_direction()

        if flip_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = flip_dir

            h, w = results['image_shape'][:2]
            # flip image
            if isinstance(results['img'], list):
                results['img'] = [
                    imflip(img, direction=flip_dir) for img in results['img']
                ]
            else:
                results['img'] = imflip(results['img'], direction=flip_dir)

            # flip bboxes
            if results.get('bbox_center', None) is not None:
                results['bbox_center'] = flip_bbox(
                    results['bbox_center'],
                    image_size=(w, h),
                    bbox_format='center',
                    direction=flip_dir)

            # flip keypoints
            if results.get('keypoints', None) is not None:
                keypoints, keypoints_visible = flip_keypoints(
                    results['keypoints'],
                    results.get('keypoints_visible', None),
                    image_size=(w, h),
                    flip_pairs=results['flip_pairs'],
                    direction=flip_dir)

                results['keypoints'] = keypoints
                results['keypoints_visible'] = keypoints_visible

        return results


@TRANSFORMS.register_module()
class TopDownRandomHalfBody(BaseTransform):
    """Data augmentation with half-body transform that keeps only the upper or
    lower body at random.

    Required Keys:

        - keypoints
        - keypoints_visible
        - upper_body_ids
        - lower_body_ids
        - input_size

    Modified Keys:

        - bbox
        - bbox_center
        - bbox_scale

    Args:
        min_total_keypoints (int): The minimum required number of total valid
            keypoints of a person to apply half-body transform. Defaults to 8
        min_half_keypoints (int): The minimum required number of valid
            half-body keypoints of a person to apply half-body transform.
            Defaults to 2
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.5
        pixel_std (float): The normalization factor to calculate `bbox_scale`.
            Defaults to 200.0
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3
    """

    def __init__(self,
                 min_total_keypoints: int = 8,
                 min_half_keypoints: int = 2,
                 padding: float = 1.5,
                 pixel_std: float = 200.0,
                 prob: float = 3) -> None:
        super().__init__()
        self.min_total_keypoints = min_total_keypoints
        self.min_half_keypoints = min_half_keypoints
        self.padding = padding
        self.pixel_std = pixel_std
        self.prob = prob

    def _get_half_body_bbox(self, keypoints: np.ndarray, input_size: Tuple,
                            half_body_ids: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Get half-body bbox center and scale of a single instance.

        Args:
            keypoints (np.ndarray): Keypoints in shape (K, C)
            input_size (tuple): The image size of the model input in
                [w, h]
            upper_body_ids (list): The list of half-body keypont indices

        Returns:
            tuple: A tuple containing half-body bbox center and scale
            - center: Center (x, y) of the bbox
            - scale: Scale (w, h) of the bbox
        """

        selected_keypoints = keypoints[half_body_ids]
        center = selected_keypoints.mean(axis=0)[:2]

        x1, y1 = selected_keypoints.min(axis=0)
        x2, y2 = selected_keypoints.max(axis=0)
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = input_size[0] / input_size[1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w, h], dtype=center.dtype)
        scale = scale * self.padding / self.pixel_std

        return center, scale

    @cache_randomness
    def _random_select_half_body(self, keypoints_visible: np.ndarray,
                                 upper_body_ids: List[int],
                                 lower_body_ids: List[int]
                                 ) -> List[Optional[List[int]]]:
        """Randomly determine whether applying half-body transform and get the
        half-body keyponit indices of each instances.

        Args:
            keypoints_visible (np.ndarray, optional): The visibility of
                keypoints in shape (N, K, 1).
            upper_body_ids (list): The list of upper body keypoint indices
            lower_body_ids (list): The list of lower body keypoint indices

        Returns:
            list[list[int] | None]: The selected half-body keypoint indices
            of each instance. ``None`` means not applying half-body transform.
        """

        half_body_ids = []

        for visible in keypoints_visible:
            if visible.sum() < self.min_total_keypoints:
                indices = None
            elif np.random.rand() > self.prob:
                indices = None
            else:
                upper_valid_ids = [
                    i for i in upper_body_ids if visible[i, 0] > 0
                ]
                lower_valid_ids = [
                    i for i in lower_body_ids if visible[i, 0] > 0
                ]

                num_upper = len(upper_valid_ids)
                num_lower = len(lower_valid_ids)

                user_upper_body = np.random.rand() < 0.5
                if (num_upper < self.min_half_keypoints
                        and num_lower < self.min_half_keypoints):
                    indices = None
                elif num_lower < self.min_half_keypoints or user_upper_body:
                    indices = upper_valid_ids.copy()
                else:
                    indices = lower_valid_ids.copy()

            half_body_ids.append(indices)

        return half_body_ids

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`TopDownHalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        half_body_ids = self._random_select_half_body(
            keypoints_visible=results['keypoints_visible'],
            upper_body_ids=results['upper_body_ids'],
            lower_body_ids=results['lower_body_ids'])

        bbox_center = []
        bbox_scale = []

        for i, indices in enumerate(half_body_ids):
            if indices is None:
                bbox_center.append(results['bbox_center'][i])
                bbox_scale.append(results['bbox_scale'][i])
            else:
                _center, _scale = self._get_half_body_bbox(
                    results['keypoints'][i], results['input_size'], indices)
                bbox_center.append(_center)
                bbox_scale.append(_scale)

        results['bbox_center'] = np.stack(bbox_center)
        results['bbox_scale'] = np.stack(bbox_scale)


@TRANSFORMS.register_module()
class TopDownRandomBboxTransform(BaseTransform):
    r"""Rnadomly shift, resize and rotate the bounding boxes.

    Required Keys:

        - bbox_center
        - bbox_scale

    Modified Keys:

        - bbox_center
        - bbox_scale

    Added Keys:
        - bbox_rotation

    Args:
        shift_factor (float): Randomly shift the bbox in range
            :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
            where :math:`dx(y) = x(y)_scale \cdot pixel_std \cdot
            shift_factor` in pixels. Defaults to 0.16
        shift_prob (float): Probability of applying random shift. Defaults to
            0.3
        scale_factor (float): Randomly resize the bbox in range
            :math:`[1 - scale_factor]
        scale_prob (float): Probability of random resizing bbox. Defaults to:
            0.5
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-2*rotate_factor, 2*rotate_factor]` in degrees. Defaults
            to 40.0
        pixel_std (float): The normalization factor to calculate `bbox_scale`.
            Defaults to 200.0
    """

    def __init__(self,
                 shift_factor: float = 0.16,
                 shift_prob: float = 0.3,
                 scale_factor: float = 0.5,
                 scale_prob: float = 1.0,
                 rotate_factor: float = 40.0,
                 rotate_prob: float = 0.6,
                 pixel_std: float = 200.) -> None:
        super().__init__()

        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob
        self.pixel_std = pixel_std

    @staticmethod
    def _truncnorm(low: float = -1.,
                   high: float = 1.,
                   size: tuple = ()) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=size).astype(np.float32)

    @cache_randomness
    def _get_transform_params(self, bbox_scale: np.ndarray) -> Tuple:
        """Get random transform parameters.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)

        Returns:
            tuple:
            - offset (np.ndarray): Offset of each bbox in shape (n, 2)
            - scale (np.ndarray): Scale factor of each bbox in shape (n, 1)
            - rotate (np.ndarray): Rotation degree of each bbox in shape
                (n, 1)
        """
        if np.random.rand() > self.prob:
            offset = np.zeros_like(bbox_scale)
        else:
            offset = np.random.uniform(-1, 1, bbox_scale.shape)
            offset = offset * self.shift_factor * bbox_scale * self.pixel_std

        num_bbox = bbox_scale.shape[0]

        # Get shift parameters
        offset = self._truncnorm(size=(num_bbox, 2))
        offset = offset * self.shift_factor * bbox_scale * self.pixel_std
        offset = np.where(
            np.random.rand(num_bbox, 1) < self.shift_prob, offset, 0.)

        # Get scaling parameters
        scale = self.truncnorm(size=(num_bbox, 1))
        scale = scale * self.scale_factor + 1.
        scale = np.where(
            np.random.rand(num_bbox, 1) < self.scale_prob, scale, 1.)

        # Get rotation parameters
        # TODO: check why use [-2, 2] truncation instead of [-1, 1]
        rotate = self.truncnorm(-2, 2, size=(num_bbox, 1))
        rotate = rotate * self.rotate_factor
        rotate = np.where(
            np.random.rand(num_bbox) < self.rotate_factor, rotate, 0.)

        return offset, scale, rotate

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`TopDownRandomBboxTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        offset, scale, rotate = self._get_transform_params(
            results['bbox_scale'])

        results['bbox_center'] += offset
        results['bbox_scale'] *= scale
        results['bbox_rotation'] = rotate

        return results


@TRANSFORMS.register_module()
class TopDownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation
        - keypoints
        - keypoints_visible

    Modified Keys:

        - img
        - keypoints

    Args:
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``
        pixel_std (float): The normalization factor to calculate `bbox_scale`.
            Defaults to 200.0

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self, use_udp: bool = False, pixel_std: float = 200.) -> None:
        super().__init__()

        self.use_udp = use_udp
        self.pixel_std = pixel_std

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`TopDownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = results['input_size']
        warp_size = (int(w), int(h))

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                results['bbox_center'],
                results['bbox_scale'],
                results['bbox_rotation'],
                output_size=(w - 1, h - 1))
        else:
            warp_mat = get_warp_matrix(
                results['bbox_center'],
                results['bbox_scale'],
                results['bbox_rotation'],
                output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        # Only transform (x, y) coordinates of the keypoints
        results['keypoints'][..., :2] = cv2.transform(
            results['keypoints'][..., :2], warp_mat)

        return results


@TRANSFORMS.register_module()
class TopDownGenerateHeatmap(BaseTransform):
    """Generate the target heatmap.

    Required Keys:

        - keypoints
        - keypoints_visible

    Added Keys:

        - heatmap
        - target_weight

    Args:
        encoding (str): Approach to encode keypoints to heatmaps. Options are
            ``'msra'`` (see `Simple Baseline`_), ``'megvii'`` (see `MSPN`_ and
            `CPN`_) and ``'udp'`` (see `UDP`_). Defaults to ``'msra'``
        sigma (float | list[float]): Gaussian sigma value(s)in ``'msra'`` and
            ``'udp'`` encoding. Defaults to 2.0
        unbiased (bool): Whether use unbiased method in ``'msra'`` encoding.
            See `Dark Pose`_ for details. Defaults to ``False``
        kernel_size (tuple | list[tuple]): The size of Gaussian kernel(s) in
            ``'megvii'`` encoding. Defaults to (11, 11)
        udp_combined_map (bool): Whether use combined map in ``udp`` encoding.
            If ``True``, the generated map is a combination of a binary
            heatmap (for classification) and an offset map (for regression).
            Otherwise, the generated map is a gaussian heatmap. Defaults to
            ``False``
        udp_valid_radius_factor (float | list[float]): The valid radius
            factor(s) for ``'udp'`` encoding with ``udp_combined_map==True``.
            Defaults to 0.0546875

    .. _`Simple Baseline`: https://arxiv.org/abs/1804.06208
    .. _`MSPN`: https://arxiv.org/abs/1901.00148
    .. _`CPN`: https://arxiv.org/abs/1711.07319
    .. _`UDP`: https://arxiv.org/abs/1911.07524
    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    def __init__(self,
                 encoding: str = 'msra',
                 sigma: Union[float, List[float]] = 2.,
                 unbiased: bool = False,
                 kernel_size: Union[Tuple, List[Tuple]] = (11, 11),
                 udp_combined_map: bool = False,
                 udp_valid_radius_factor: Union[float, List[float]] = 0.0546875
                 ) -> None:
        super().__init__()

        self.encoding = encoding
        self.sigma = sigma
        self.unbiased = unbiased
        self.kernel_size = kernel_size
        self.udp_combined_map = udp_combined_map
        self.udp_valid_raidus_factor = udp_valid_radius_factor
