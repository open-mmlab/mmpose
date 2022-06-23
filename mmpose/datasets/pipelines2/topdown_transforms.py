# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv.image import imflip
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine import is_list_of, is_seq_of
from scipy.stats import truncnorm

from mmpose.core.bbox import (bbox_xywh2cs, flip_bbox, get_udp_warp_matrix,
                              get_warp_matrix)
from mmpose.core.keypoint import (flip_keypoints, generate_megvii_heatmap,
                                  generate_msra_heatmap, generate_udp_heatmap)
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TopDownGetBboxCenterScale(BaseTransform):
    """Convert bboxes from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
    """

    def __init__(self, padding: float = 1.25) -> None:
        super().__init__()

        self.padding = padding

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
            center, scale = bbox_xywh2cs(bbox, padding=self.padding)

            results['bbox_center'] = center
            results['bbox_scale'] = scale

        return results


@TRANSFORMS.register_module()
class TopDownRandomFlip(BaseTransform):
    """Randomly flip the image, bbox and keypoints.

    Required Keys:

        - img
        - img_shape
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

            h, w = results['img_shape'][:2]
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
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3
    """

    def __init__(self,
                 min_total_keypoints: int = 8,
                 min_half_keypoints: int = 2,
                 padding: float = 1.5,
                 prob: float = 0.3) -> None:
        super().__init__()
        self.min_total_keypoints = min_total_keypoints
        self.min_half_keypoints = min_half_keypoints
        self.padding = padding
        self.prob = prob

    def _get_half_body_bbox(self, keypoints: np.ndarray,
                            half_body_ids: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Get half-body bbox center and scale of a single instance.

        Args:
            keypoints (np.ndarray): Keypoints in shape (K, C)
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
        scale = np.array([w, h], dtype=center.dtype) * self.padding

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
                    results['keypoints'][i], indices)
                bbox_center.append(_center)
                bbox_scale.append(_scale)

        results['bbox_center'] = np.stack(bbox_center)
        results['bbox_scale'] = np.stack(bbox_scale)
        return results


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
            where :math:`dx(y) = x(y)_scale \cdot shift_factor` in pixels.
            Defaults to 0.16
        shift_prob (float): Probability of applying random shift. Defaults to
            0.3
        scale_factor (float): Randomly resize the bbox in range
            :math:`[1 - scale_factor, 1 + scale_factor]`. Defaults to 0.5
        scale_prob (float): Probability of random resizing bbox. Defaults to:
            0.5
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-2*rotate_factor, 2*rotate_factor]` in degrees. Defaults
            to 40.0
    """

    def __init__(self,
                 shift_factor: float = 0.16,
                 shift_prob: float = 0.3,
                 scale_factor: float = 0.5,
                 scale_prob: float = 1.0,
                 rotate_factor: float = 40.0,
                 rotate_prob: float = 0.6) -> None:
        super().__init__()

        assert 0. < scale_factor < 1.0

        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob

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
        num_bbox = bbox_scale.shape[0]

        # Get shift parameters
        offset = self._truncnorm(size=(num_bbox, 2))
        offset = offset * self.shift_factor * bbox_scale
        offset = np.where(
            np.random.rand(num_bbox, 1) < self.shift_prob, offset, 0.)

        # Get scaling parameters
        scale = self._truncnorm(size=(num_bbox, 1))
        scale = scale * self.scale_factor + 1.
        scale = np.where(
            np.random.rand(num_bbox, 1) < self.scale_prob, scale, 1.)

        # Get rotation parameters
        # TODO: check why use [-2, 2] truncation instead of [-1, 1]
        rotate = self._truncnorm(-2, 2, size=(num_bbox, 1))
        rotate = rotate * self.rotate_factor
        rotate = np.where(
            np.random.rand(num_bbox) < self.rotate_prob, rotate, 0.)

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

    Added Keys:

        - input_size

    Args:
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w`` to ``h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """
        w, h = bbox_scale[:, 0], bbox_scale[:, 1]
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`TopDownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        center = results['bbox_center']
        scale = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)
        rot = results['bbox_rotation']

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w - 1, h - 1))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        # Only transform (x, y) coordinates
        results['keypoints'][..., :2] = cv2.transform(
            results['keypoints'][..., :2], warp_mat)

        results['input_size'] = (w, h)

        return results


@TRANSFORMS.register_module()
class TopDownGenerateHeatmap(BaseTransform):
    r"""Generate the target heatmap of the keypoints.

    Required Keys:

        - keypoints
        - keypoints_visible
        - keypoint_weights

    Added Keys:

        - target
        - target_weight

    Args:
        heatmap_size (tuple): The heatmap size in [w, h]
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
        udp_radius_factor (float | list[float]): The radius factor(s) for
            ``'udp'`` encoding with ``udp_combined_map==True``. The keypoint
            radius in the heatmap is calculated as
            :math:`factor \times max(w, h)` in pixels. Defaults to 0.0546875
            (equivalent to 3.5 in 48x64 heatmap and 5.25 in 72x96 heatmap)
        use_meta_keypoint_weight (bool): Whether use the keypoint weights from
            the dataset meta information. Defaults to ``False``

    .. _`Simple Baseline`: https://arxiv.org/abs/1804.06208
    .. _`MSPN`: https://arxiv.org/abs/1901.00148
    .. _`CPN`: https://arxiv.org/abs/1711.07319
    .. _`UDP`: https://arxiv.org/abs/1911.07524
    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    def __init__(self,
                 heatmap_size: Tuple[int, int],
                 encoding: str = 'msra',
                 sigma: Union[float, List[float]] = 2.,
                 unbiased: bool = False,
                 kernel_size: Union[Tuple, List[Tuple]] = (11, 11),
                 udp_combined_map: bool = False,
                 udp_radius_factor: Union[float, List[float]] = 0.0546875,
                 use_meta_keypoint_weight: bool = False) -> None:
        super().__init__()

        encoding_options = ['msra', 'megvii', 'udp']

        assert encoding.lower() in encoding_options, (
            f'Invalid encoding type "{encoding}"'
            f'Options are {encoding_options}')

        assert is_seq_of(heatmap_size, int) and len(heatmap_size, 2), (
            'heatmap_size should be a tuple of integers as (w, h), got'
            f'invalid value {heatmap_size}')

        self.heatmap_size = heatmap_size
        self.encoding = encoding.lower()
        self.sigma = sigma
        self.unbiased = unbiased
        self.kernel_size = kernel_size
        self.udp_combined_map = udp_combined_map
        self.udp_radius_factor = udp_radius_factor
        self.use_meta_keypoint_weight = use_meta_keypoint_weight

        # get heatmap encoder and its arguments
        self.encoder, self.encoder_args = self._get_encoder()

    def _get_encoder(self):
        """Get heatmap generation function and arguments.

        Returns:
            tuple:
            - encoder [callable]: The heatmap generation function
            - encoder_args [dict | list[dict]]: The keyword arguments of
                ``encoder``. A list means to generate multi-level
                heatmaps where each element is the keyword arguments of
                one level
        """

        if self.encoding == 'msra':
            encoder = generate_msra_heatmap

            if isinstance(self.sigma, list):
                encoder_args = [{
                    'sigma': sigma,
                    'unbiased': self.unbiased
                } for sigma in self.sigma]
            else:
                encoder_args = {'sigma': self.sigma, 'unbiased': self.unbiased}

        elif self.encoding == 'megvii':
            encoder = generate_megvii_heatmap

            if is_list_of(self.kernel_size, (list, tuple)):
                encoder_args = [{
                    'kernel_size': kernel_size
                } for kernel_size in self.kernel_size]
            else:
                encoder_args = {'kernel_size': self.kernel_size}

        elif self.encoding == 'udp':
            encoder = generate_udp_heatmap

            if self.udp_combined_map:
                factor = self.udp_radius_factor
            else:
                factor = self.sigma

            if isinstance(factor, list):
                encoder_args = [{
                    'factor': _factor,
                    'combined_map': self.udp_combined_map
                } for _factor in factor]
            else:
                encoder_args = {
                    'factor': factor,
                    'combined_map': self.udp_combined_map
                }

        else:
            raise ValueError(f'Invalid encoding type {self.encoding}')

        return encoder, encoder_args

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`TopDownGenerateHeatmap`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        # TODO: support multi-instance heatmap encoding
        assert results['keypoints'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. '
            f'Got invalid shape of keypoints {results["keypoints"].shape}.')

        keypoints = results['keypoints'][0]
        keypoints_visible = results['keypoints_visible'][0]

        encoder_args = deepcopy(self.encoder_args)

        if isinstance(encoder_args, list):
            # multi-level heatmap
            heatmaps = []
            keypoint_weights = []

            for args in encoder_args:
                _heatmap, _keypoint_weight = self.encoder(
                    keypoints=keypoints[0],
                    keypoints_visible=keypoints_visible,
                    image_size=results['input_size'],
                    heatmap_size=self.heatmap_size,
                    **args)

                heatmaps.append(_heatmap)
                keypoint_weights.append(_keypoint_weight)

            heatmap = np.stack(heatmaps)
            keypoint_weight = np.stack(keypoint_weights)

        else:
            # single-level heatmap
            heatmap, keypoint_weight = self.encoder(
                keypoints=keypoints[0],
                keypoints_visible=keypoints_visible,
                image_size=results['input_size'],
                heatmap_size=self.heatmap_size,
                **encoder_args)

        # multiply meta keypoint weight
        if self.use_meta_keypoint_weight:
            keypoint_weight *= results['keypoint_weights'][:, None]

        results['target'] = heatmap
        results['target_weight'] = keypoint_weight

        return results


@TRANSFORMS.register_module()
class TopDownGenerateRegressionLabel(BaseTransform):
    """Generate the target regression label of the keypoints.

    Required Keys:

        - keypoints
        - keypoints_visible
        - image_size

    Added Keys:

        - target
        - target_weight

    Args:
        use_meta_keypoint_weight (bool): Whether use the keypoint weights from
            the dataset meta information. Defaults to ``False``
    """

    def __init__(self, use_meta_keypoint_weight: bool = False) -> None:
        super().__init__()
        self.use_meta_keypoint_weight = use_meta_keypoint_weight

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`TopDownGenerateRegressionLabel`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        # TODO: support multi-instance regression label encoding
        assert results['keypoints'].shape[0] == 1, (
            'Top-down regression only supports single instance. '
            f'Got invalid shape of keypoints {results["keypoints"].shape}.')
        keypoints = results['keypoints'][0]
        keypoints_visible = results['keypoints_visible'][0]

        w, h = results['input_size']
        valid = ((keypoints >= 0) &
                 (keypoints <= [w - 1, h - 1]) & keypoints_visible > 0.5).all(
                     axis=1)

        target = keypoints / [w, h]
        target_weight = np.where(valid, 1., 0.).astype(np.float32)

        # multiply meta keypoint weight
        if self.use_meta_keypoint_weight:
            target_weight *= results['keypoint_weights'][:, None]

        results['target'] = target
        results['target_weight'] = target_weight

        return results
