# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.image import imflip
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine import is_list_of
from scipy.stats import truncnorm

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import bbox_xyxy2cs, flip_bbox
from mmpose.structures.keypoint import flip_keypoints

try:
    import albumentations
except ImportError:
    albumentations = None

Number = Union[int, float]


@TRANSFORMS.register_module()
class GetBBoxCenterScale(BaseTransform):
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
        """The transform function of :class:`GetBBoxCenterScale`.

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
            center, scale = bbox_xyxy2cs(bbox, padding=self.padding)

            results['bbox_center'] = center
            results['bbox_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(padding={self.padding})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Randomly flip the image, bbox and keypoints.

    Required Keys:

        - img
        - img_shape
        - flip_pairs
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)

    Modified Keys:

        - img
        - bbox (optional)
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

            h, w = results['img_shape']
            # flip image
            if isinstance(results['img'], list):
                results['img'] = [
                    imflip(img, direction=flip_dir) for img in results['img']
                ]
            else:
                results['img'] = imflip(results['img'], direction=flip_dir)

            # flip bboxes
            if results.get('bbox', None) is not None:
                results['bbox'] = flip_bbox(
                    results['bbox'],
                    image_size=(w, h),
                    bbox_format='xyxy',
                    direction=flip_dir)

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

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'
        return repr_str


@TRANSFORMS.register_module()
class RandomHalfBody(BaseTransform):
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
            keypoints (np.ndarray): Keypoints in shape (K, D)
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
                upper_valid_ids = [i for i in upper_body_ids if visible[i] > 0]
                lower_valid_ids = [i for i in lower_body_ids if visible[i] > 0]

                num_upper = len(upper_valid_ids)
                num_lower = len(lower_valid_ids)

                select_upper = np.random.rand() < 0.5
                if (num_upper < self.min_half_keypoints
                        and num_lower < self.min_half_keypoints):
                    indices = None
                elif num_lower < self.min_half_keypoints:
                    indices = upper_valid_ids
                elif num_upper < self.min_half_keypoints:
                    indices = lower_valid_ids
                else:
                    indices = (
                        upper_body_ids if select_upper else lower_body_ids)

            half_body_ids.append(indices)

        return half_body_ids

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`HalfBodyTransform`.

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

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(min_total_keypoints={self.min_total_keypoints}, '
        repr_str += f'min_half_keypoints={self.min_half_keypoints}, '
        repr_str += f'padding={self.padding}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomBBoxTransformV0(BaseTransform):

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
        offset = np.random.uniform(-1, 1, size=(num_bbox, 2))
        offset = offset * self.shift_factor * bbox_scale
        offset = np.where(
            np.random.rand(num_bbox, 1) < self.shift_prob, offset, 0.)

        # Get scaling parameters
        scale = np.clip(
            np.random.randn(num_bbox, 1) * self.scale_factor,
            -self.scale_factor, self.scale_factor)
        scale += 1
        scale = np.where(
            np.random.rand(num_bbox, 1) < self.scale_prob, scale, 1.)

        # Get rotation parameters
        rotate = np.clip(
            np.random.randn(num_bbox) * self.rotate_factor,
            -self.rotate_factor * 2, self.rotate_factor * 2)
        rotate = np.where(
            np.random.rand(num_bbox) < self.rotate_prob, rotate, 0.)

        return offset, scale, rotate

    def transform(self, results: Dict) -> Optional[dict]:
        offset, scale, rotate = self._get_transform_params(
            results['bbox_scale'])

        results['bbox_center'] += offset
        results['bbox_scale'] *= scale
        results['bbox_rotation'] = rotate

        return results


@TRANSFORMS.register_module()
class RandomBBoxTransform(BaseTransform):
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
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-2*rotate_factor, 2*rotate_factor]` in degrees. Defaults
            to 40.0
        rotate_prob (float): Probability of applying random rotation. Defaults
            to 0.6
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
        rotate = self._truncnorm(-2, 2, size=(num_bbox, ))
        rotate = rotate * self.rotate_factor
        rotate = np.where(
            np.random.rand(num_bbox) < self.rotate_prob, rotate, 0.)

        return offset, scale, rotate

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`RandomBBoxTransform`.

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

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(shift_prob={self.shift_prob}, '
        repr_str += f'shift_factor={self.shift_factor}, '
        repr_str += f'scale_prob={self.scale_prob}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'rotate_prob={self.rotate_prob}, '
        repr_str += f'rotate_factor={self.rotate_factor})'
        return repr_str


@TRANSFORMS.register_module()
@avoid_cache_randomness
class Albumentation(BaseTransform):
    """Albumentation augmentation (pixel-level transforms only).

    Adds custom pixel-level transformations from Albumentations library.
    Please visit `https://albumentations.ai/docs/`
    to get more information.

    Note: we only support pixel-level transforms.
    Please visit `https://github.com/albumentations-team/`
    `albumentations#pixel-level-transforms`
    to get more information about pixel-level transforms.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        transforms (List[dict]): A list of Albumentation transforms.
            An example of ``transforms`` is as followed:
            .. code-block:: python

                [
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.1),
                ]
        keymap (dict | None): key mapping from ``input key`` to
            ``albumentation-style key``.
            Defaults to None, which will use {'img': 'image'}.
    """

    def __init__(self,
                 transforms: List[dict],
                 keymap: Optional[dict] = None) -> None:
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            albumentations.BasicTransform: The constructed transform object
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            if not hasattr(albumentations.augmentations.transforms, obj_type):
                warnings.warn('{obj_type} is not pixel-level transformations. '
                              'Please use with caution.')
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict) -> dict:
        """Dictionary mapper.

        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): key mapping like {'old_key': 'new_key'}.

        Returns:
            dict: new dict.
        """

        updated_dict = {keymap.get(k, k): v for k, v in d.items()}
        return updated_dict

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`Albumentation` to apply
        albumentations transforms.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): Result dict from the data pipeline.

        Return:
            dict: updated result dict.
        """
        # map result dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # Apply albumentations transforms
        results = self.aug(**results)
        # map result dict back to the original format
        results = self.mapper(results, self.keymap_back)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@TRANSFORMS.register_module()
class PhotometricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[Number] = (0.5, 1.5),
                 saturation_range: Sequence[Number] = (0.5, 1.5),
                 hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @cache_randomness
    def _random_flags(self) -> Sequence[Number]:
        """Generate the random flags for subsequent transforms.

        Returns:
            Sequence[Number]: a sequence of numbers that indicate whether to
                do the corresponding transforms.
        """
        # contrast_mode == 0 --> do random contrast first
        # contrast_mode == 1 --> do random contrast last
        contrast_mode = np.random.randint(2)
        # whether to apply brightness distortion
        brightness_flag = np.random.randint(2)
        # whether to apply contrast distortion
        contrast_flag = np.random.randint(2)
        # the mode to convert color from BGR to HSV
        hsv_mode = np.random.randint(4)
        # whether to apply channel swap
        swap_flag = np.random.randint(2)

        # the beta in `self._convert` to be added to image array
        # in brightness distortion
        brightness_beta = np.random.uniform(-self.brightness_delta,
                                            self.brightness_delta)
        # the alpha in `self._convert` to be multiplied to image array
        # in contrast distortion
        contrast_alpha = np.random.uniform(self.contrast_lower,
                                           self.contrast_upper)
        # the alpha in `self._convert` to be multiplied to image array
        # in saturation distortion to hsv-formatted img
        saturation_alpha = np.random.uniform(self.saturation_lower,
                                             self.saturation_upper)
        # delta of hue to add to image array in hue distortion
        hue_delta = np.random.randint(-self.hue_delta, self.hue_delta)
        # the random permutation of channel order
        swap_channel_order = np.random.permutation(3)

        return (contrast_mode, brightness_flag, contrast_flag, hsv_mode,
                swap_flag, brightness_beta, contrast_alpha, saturation_alpha,
                hue_delta, swap_channel_order)

    def _convert(self,
                 img: np.ndarray,
                 alpha: float = 1,
                 beta: float = 0) -> np.ndarray:
        """Multiple with alpha and add beta with clip.

        Args:
            img (np.ndarray): The image array.
            alpha (float): The random multiplier.
            beta (float): The random offset.

        Returns:
            np.ndarray: The updated image array.
        """
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`PhotometricDistortion` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        assert 'img' in results, '`img` is not found in results'
        img = results['img']

        (contrast_mode, brightness_flag, contrast_flag, hsv_mode, swap_flag,
         brightness_beta, contrast_alpha, saturation_alpha, hue_delta,
         swap_channel_order) = self._random_flags()

        # random brightness distortion
        if brightness_flag:
            img = self._convert(img, beta=brightness_beta)

        # contrast_mode == 0 --> do random contrast first
        # contrast_mode == 1 --> do random contrast last
        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)

        if hsv_mode:
            # random saturation/hue distortion
            img = mmcv.bgr2hsv(img)
            if hsv_mode == 1 or hsv_mode == 3:
                # apply saturation distortion to hsv-formatted img
                img[:, :, 1] = self._convert(
                    img[:, :, 1], alpha=saturation_alpha)
            if hsv_mode == 2 or hsv_mode == 3:
                # apply hue distortion to hsv-formatted img
                img[:, :, 0] = img[:, :, 0].astype(int) + hue_delta
            img = mmcv.hsv2bgr(img)

        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)

        # randomly swap channels
        if swap_flag:
            img = img[..., swap_channel_order]

        results['img'] = img
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str
