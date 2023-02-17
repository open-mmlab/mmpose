# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import xtcocotools.mask as cocomask
from mmcv.image import imflip_, imresize
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from scipy.stats import truncnorm

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix


@TRANSFORMS.register_module()
class BottomupGetHeatmapMask(BaseTransform):
    """Generate the mask of valid regions from the segmentation annotation.

    Required Keys:

        - img_shape
        - invalid_segs (optional)
        - warp_mat (optional)
        - flip (optional)
        - flip_direction (optional)
        - heatmaps (optional)

    Added Keys:

        - heatmap_mask
    """

    def _segs_to_mask(self, segs: list, img_shape: Tuple[int,
                                                         int]) -> np.ndarray:
        """Calculate mask from object segmentations.

        Args:
            segs (List): The object segmentation annotations in COCO format
            img_shape (Tuple): The image shape in (h, w)

        Returns:
            np.ndarray: The binary object mask in size (h, w), where the
            object pixels are 1 and background pixels are 0
        """

        # RLE is a simple yet efficient format for storing binary masks.
        # details can be found at `COCO tools <https://github.com/
        # cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/
        # mask.py>`__
        rles = []
        for seg in segs:
            rle = cocomask.frPyObjects(seg, img_shape[0], img_shape[1])
            if isinstance(rle, list):
                # For non-crowded objects (e.g. human with no visible
                # keypoints), the results is a list of rles
                rles.extend(rle)
            else:
                # For crowded objects, the result is a single rle
                rles.append(rle)

        if rles:
            mask = cocomask.decode(cocomask.merge(rles))
        else:
            mask = np.zeros(img_shape, dtype=np.uint8)

        return mask

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BottomupGetHeatmapMask` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        invalid_segs = results.get('invalid_segs', [])
        img_shape = results['img_shape']  # (img_h, img_w)
        input_size = results['input_size']

        # Calculate the mask of the valid region by negating the segmentation
        # mask of invalid objects
        mask = 1 - self._segs_to_mask(invalid_segs, img_shape)

        # Apply an affine transform to the mask if the image has been
        # transformed
        if 'warp_mat' in results:
            warp_mat = results['warp_mat']

            mask = mask.astype(np.float32)
            mask = cv2.warpAffine(
                mask, warp_mat, input_size, flags=cv2.INTER_LINEAR)

        # Flip the mask if the image has been flipped
        if results.get('flip', False):
            flip_dir = results['flip_direction']
            if flip_dir is not None:
                mask = imflip_(mask, flip_dir)

        # Resize the mask to the same size of heatmaps
        if 'heatmaps' in results:
            heatmaps = results['heatmaps']
            if isinstance(heatmaps, list):
                # Multi-level heatmaps
                heatmap_mask = []
                for hm in results['heatmaps']:
                    h, w = hm.shape[1:3]
                    _mask = imresize(
                        mask, size=(w, h), interpolation='bilinear')
                    heatmap_mask.append(_mask)
            else:
                h, w = heatmaps.shape[1:3]
                heatmap_mask = imresize(
                    mask, size=(w, h), interpolation='bilinear')
        else:
            heatmap_mask = mask

        # Binarize the mask(s)
        if isinstance(heatmap_mask, list):
            results['heatmap_mask'] = [hm > 0.5 for hm in heatmap_mask]
        else:
            results['heatmap_mask'] = heatmap_mask > 0.5

        return results


@TRANSFORMS.register_module()
class BottomupRandomAffine(BaseTransform):
    r"""Randomly shift, resize and rotate the image.

    Required Keys:

        - img
        - img_shape
        - keypoints (optional)

    Modified Keys:

        - img
        - keypoints (optional)

    Added Keys:

        - input_size
        - warp_mat

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]
        shift_factor (float): Randomly shift the image in range
            :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
            where :math:`dx(y) = img_w(h) \cdot shift_factor` in pixels.
            Defaults to 0.2
        shift_prob (float): Probability of applying random shift. Defaults to
            1.0
        scale_factor (Tuple[float, float]): Randomly resize the image in range
            :math:`[scale_factor[0], scale_factor[1]]`. Defaults to
            (0.75, 1.5)
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
        scale_type (str): wrt ``long`` or ``short`` length of the image.
            Defaults to ``short``
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-rotate_factor, rotate_factor]` in degrees. Defaults
            to 40.0
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 shift_factor: float = 0.2,
                 shift_prob: float = 1.,
                 scale_factor: Tuple[float, float] = (0.75, 1.5),
                 scale_prob: float = 1.,
                 scale_type: str = 'short',
                 rotate_factor: float = 30.,
                 rotate_prob: float = 1,
                 use_udp: bool = False) -> None:
        super().__init__()

        self.input_size = input_size
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.scale_type = scale_type
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob
        self.use_udp = use_udp

    @staticmethod
    def _truncnorm(low: float = -1.,
                   high: float = 1.,
                   size: tuple = ()) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=size).astype(np.float32)

    def _fix_aspect_ratio(self, scale: np.ndarray, aspect_ratio: float):
        """Extend the scale to match the given aspect ratio.

        Args:
            scale (np.ndarray): The image scale (w, h) in shape (2, )
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.ndarray: The reshaped image scale in (2, )
        """
        w, h = scale
        if w > h * aspect_ratio:
            if self.scale_type == 'long':
                _w, _h = w, w / aspect_ratio
            elif self.scale_type == 'short':
                _w, _h = h * aspect_ratio, h
            else:
                raise ValueError(f'Unknown scale type: {self.scale_type}')
        else:
            if self.scale_type == 'short':
                _w, _h = w, w / aspect_ratio
            elif self.scale_type == 'long':
                _w, _h = h * aspect_ratio, h
            else:
                raise ValueError(f'Unknown scale type: {self.scale_type}')
        return np.array([_w, _h], dtype=scale.dtype)

    @cache_randomness
    def _get_transform_params(self) -> Tuple:
        """Get random transform parameters.

        Returns:
            tuple:
            - offset (np.ndarray): Image offset rate in shape (2, )
            - scale (np.ndarray): Image scaling rate factor in shape (1, )
            - rotate (np.ndarray): Image rotation degree in shape (1, )
        """
        # get offset
        if np.random.rand() < self.shift_prob:
            offset = self._truncnorm(size=(2, )) * self.shift_factor
        else:
            offset = np.zeros((2, ), dtype=np.float32)

        # get scale
        if np.random.rand() < self.scale_prob:
            scale_min, scale_max = self.scale_factor
            scale = scale_min + (scale_max - scale_min) * (
                self._truncnorm(size=(1, )) + 1) / 2
        else:
            scale = np.ones(1, dtype=np.float32)

        # get rotation
        if np.random.rand() < self.rotate_prob:
            rotate = self._truncnorm() * self.rotate_factor
        else:
            rotate = 0

        return offset, scale, rotate

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BottomupRandomAffine` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img_h, img_w = results['img_shape']
        w, h = self.input_size

        offset_rate, scale_rate, rotate = self._get_transform_params()
        offset = offset_rate * [img_w, img_h]
        scale = scale_rate * [img_w, img_h]
        # adjust the scale to match the target aspect ratio
        scale = self._fix_aspect_ratio(scale, aspect_ratio=w / h)

        if self.use_udp:
            center = np.array([(img_w - 1.0) / 2, (img_h - 1.0) / 2],
                              dtype=np.float32)
            warp_mat = get_udp_warp_matrix(
                center=center + offset,
                scale=scale,
                rot=rotate,
                output_size=(w, h))
        else:
            center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
            warp_mat = get_warp_matrix(
                center=center + offset,
                scale=scale,
                rot=rotate,
                output_size=(w, h))

        # warp image and keypoints
        results['img'] = cv2.warpAffine(
            results['img'], warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)

        if 'keypoints' in results:
            # Only transform (x, y) coordinates
            results['keypoints'][..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)

        if 'bbox' in results:
            bbox = np.tile(results['bbox'], 2).reshape(-1, 4, 2)
            # corner order: left_top, left_bottom, right_top, right_bottom
            bbox[:, 1:3, 0] = bbox[:, 0:2, 0]
            results['bbox'] = cv2.transform(bbox, warp_mat).reshape(-1, 8)

        results['input_size'] = self.input_size
        results['warp_mat'] = warp_mat

        return results


@TRANSFORMS.register_module()
class BottomupResize(BaseTransform):
    """Resize the image to the input size of the model. Optionally, the image
    can be resized to multiple sizes to build a image pyramid for multi-scale
    inference.

    Required Keys:

        - img
        - ori_shape

    Modified Keys:

        - img
        - img_shape

    Added Keys:

        - input_size
        - warp_mat
        - aug_scale

    Args:
        input_size (Tuple[int, int]): The input size of the model in [w, h].
            Note that the actually size of the resized image will be affected
            by ``resize_mode`` and ``size_factor``, thus may not exactly equals
            to the ``input_size``
        aug_scales (List[float], optional): The extra input scales for
            multi-scale testing. If given, the input image will be resized
            to different scales to build a image pyramid. And heatmaps from
            all scales will be aggregated to make final prediction. Defaults
            to ``None``
        size_factor (int): The actual input size will be ceiled to
                a multiple of the `size_factor` value at both sides.
                Defaults to 16
        resize_mode (str): The method to resize the image to the input size.
            Options are:

                - ``'fit'``: The image will be resized according to the
                    relatively longer side with the aspect ratio kept. The
                    resized image will entirely fits into the range of the
                    input size
                - ``'expand'``: The image will be resized according to the
                    relatively shorter side with the aspect ratio kept. The
                    resized image will exceed the given input size at the
                    longer side
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 aug_scales: Optional[List[float]] = None,
                 size_factor: int = 32,
                 resize_mode: str = 'fit',
                 use_udp: bool = False):
        super().__init__()

        self.input_size = input_size
        self.aug_scales = aug_scales
        self.resize_mode = resize_mode
        self.size_factor = size_factor
        self.use_udp = use_udp

    @staticmethod
    def _ceil_to_multiple(size: Tuple[int, int], base: int):
        """Ceil the given size (tuple of [w, h]) to a multiple of the base."""
        return tuple(int(np.ceil(s / base) * base) for s in size)

    def _get_input_size(self, img_size: Tuple[int, int],
                        input_size: Tuple[int, int]) -> Tuple:
        """Calculate the actual input size (which the original image will be
        resized to) and the padded input size (which the resized image will be
        padded to, or which is the size of the model input).

        Args:
            img_size (Tuple[int, int]): The original image size in [w, h]
            input_size (Tuple[int, int]): The expected input size in [w, h]

        Returns:
            tuple:
            - actual_input_size (Tuple[int, int]): The target size to resize
                the image
            - padded_input_size (Tuple[int, int]): The target size to generate
                the model input which will contain the resized image
        """
        img_w, img_h = img_size
        ratio = img_w / img_h

        if self.resize_mode == 'fit':
            padded_input_size = self._ceil_to_multiple(input_size,
                                                       self.size_factor)
            if padded_input_size != input_size:
                raise ValueError(
                    'When ``resize_mode==\'fit\', the input size (height and'
                    ' width) should be mulitples of the size_factor('
                    f'{self.size_factor}) at all scales. Got invalid input '
                    f'size {input_size}.')

            pad_w, pad_h = padded_input_size
            rsz_w = min(pad_w, pad_h * ratio)
            rsz_h = min(pad_h, pad_w / ratio)
            actual_input_size = (rsz_w, rsz_h)

        elif self.resize_mode == 'expand':
            _padded_input_size = self._ceil_to_multiple(
                input_size, self.size_factor)
            pad_w, pad_h = _padded_input_size
            rsz_w = max(pad_w, pad_h * ratio)
            rsz_h = max(pad_h, pad_w / ratio)

            actual_input_size = (rsz_w, rsz_h)
            padded_input_size = self._ceil_to_multiple(actual_input_size,
                                                       self.size_factor)

        else:
            raise ValueError(f'Invalid resize mode {self.resize_mode}')

        return actual_input_size, padded_input_size

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BottomupResize` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        img_h, img_w = results['ori_shape']
        w, h = self.input_size

        input_sizes = [(w, h)]
        if self.aug_scales:
            input_sizes += [(int(w * s), int(h * s)) for s in self.aug_scales]

        imgs = []
        for i, (_w, _h) in enumerate(input_sizes):

            actual_input_size, padded_input_size = self._get_input_size(
                img_size=(img_w, img_h), input_size=(_w, _h))

            if self.use_udp:
                center = np.array([(img_w - 1.0) / 2, (img_h - 1.0) / 2],
                                  dtype=np.float32)
                scale = np.array([img_w, img_h], dtype=np.float32)
                warp_mat = get_udp_warp_matrix(
                    center=center,
                    scale=scale,
                    rot=0,
                    output_size=actual_input_size)
            else:
                center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
                scale = np.array([
                    img_w * padded_input_size[0] / actual_input_size[0],
                    img_h * padded_input_size[1] / actual_input_size[1]
                ],
                                 dtype=np.float32)
                warp_mat = get_warp_matrix(
                    center=center,
                    scale=scale,
                    rot=0,
                    output_size=padded_input_size)

            _img = cv2.warpAffine(
                img, warp_mat, padded_input_size, flags=cv2.INTER_LINEAR)

            imgs.append(_img)

            # Store the transform information w.r.t. the main input size
            if i == 0:
                results['img_shape'] = padded_input_size[::-1]
                results['input_center'] = center
                results['input_scale'] = scale
                results['input_size'] = padded_input_size

        if self.aug_scales:
            results['img'] = imgs
            results['aug_scales'] = self.aug_scales
        else:
            results['img'] = imgs[0]
            results['aug_scale'] = None

        return results
