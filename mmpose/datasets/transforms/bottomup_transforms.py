# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import xtcocotools.mask as cocomask
from mmcv.image import imflip_, imresize
from mmcv.image.geometric import imrescale
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from scipy.stats import truncnorm

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import (bbox_clip_border, bbox_corner2xyxy,
                                    bbox_xyxy2corner, get_pers_warp_matrix,
                                    get_udp_warp_matrix, get_warp_matrix)
from mmpose.structures.keypoint import keypoint_clip_border


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

    def __init__(self, get_invalid: bool = False):
        super().__init__()
        self.get_invalid = get_invalid

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
        mask = self._segs_to_mask(invalid_segs, img_shape)

        if not self.get_invalid:
            # Calculate the mask of the valid region by negating the
            # segmentation mask of invalid objects
            mask = np.logical_not(mask)

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
                 input_size: Optional[Tuple[int, int]] = None,
                 shift_factor: float = 0.2,
                 shift_prob: float = 1.,
                 scale_factor: Tuple[float, float] = (0.75, 1.5),
                 scale_prob: float = 1.,
                 scale_type: str = 'short',
                 rotate_factor: float = 30.,
                 rotate_prob: float = 1,
                 shear_factor: float = 2.0,
                 shear_prob: float = 1.0,
                 use_udp: bool = False,
                 pad_val: Union[float, Tuple[float]] = 0,
                 border: Tuple[int, int] = (0, 0),
                 distribution='trunc_norm',
                 transform_mode='affine',
                 bbox_keep_corner: bool = True,
                 clip_border: bool = False) -> None:
        super().__init__()

        assert transform_mode in ('affine', 'affine_udp', 'perspective'), \
            f'the argument transform_mode should be either \'affine\', ' \
            f'\'affine_udp\' or \'perspective\', but got \'{transform_mode}\''

        self.input_size = input_size
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.scale_type = scale_type
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob
        self.shear_factor = shear_factor
        self.shear_prob = shear_prob

        self.use_udp = use_udp
        self.distribution = distribution
        self.clip_border = clip_border
        self.bbox_keep_corner = bbox_keep_corner

        self.transform_mode = transform_mode

        if isinstance(pad_val, (int, float)):
            pad_val = (pad_val, pad_val, pad_val)

        if 'affine' in transform_mode:
            self._transform = partial(
                cv2.warpAffine, flags=cv2.INTER_LINEAR, borderValue=pad_val)
        else:
            self._transform = partial(cv2.warpPerspective, borderValue=pad_val)

    def _random(self,
                low: float = -1.,
                high: float = 1.,
                size: tuple = ()) -> np.ndarray:
        if self.distribution == 'trunc_norm':
            """Sample from a truncated normal distribution."""
            return truncnorm.rvs(low, high, size=size).astype(np.float32)
        elif self.distribution == 'uniform':
            x = np.random.rand(*size)
            return x * (high - low) + low
        else:
            raise ValueError(f'the argument `distribution` should be either'
                             f'\'trunc_norn\' or \'uniform\', but got '
                             f'{self.distribution}.')

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
            offset = self._random(size=(2, )) * self.shift_factor
        else:
            offset = np.zeros((2, ), dtype=np.float32)

        # get scale
        if np.random.rand() < self.scale_prob:
            scale_min, scale_max = self.scale_factor
            scale = scale_min + (scale_max - scale_min) * (
                self._random(size=(1, )) + 1) / 2
        else:
            scale = np.ones(1, dtype=np.float32)

        # get rotation
        if np.random.rand() < self.rotate_prob:
            rotate = self._random() * self.rotate_factor
        else:
            rotate = 0

        # get shear
        if 'perspective' in self.transform_mode and np.random.rand(
        ) < self.shear_prob:
            shear = self._random(size=(2, )) * self.shear_factor
        else:
            shear = np.zeros((2, ), dtype=np.float32)

        return offset, scale, rotate, shear

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BottomupRandomAffine` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img_h, img_w = results['img_shape'][:2]
        w, h = self.input_size

        offset_rate, scale_rate, rotate, shear = self._get_transform_params()

        if 'affine' in self.transform_mode:
            offset = offset_rate * [img_w, img_h]
            scale = scale_rate * [img_w, img_h]
            # adjust the scale to match the target aspect ratio
            scale = self._fix_aspect_ratio(scale, aspect_ratio=w / h)

            if self.transform_mode == 'affine_udp':
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

        else:
            offset = offset_rate * [w, h]
            center = np.array([w / 2, h / 2], dtype=np.float32)
            warp_mat = get_pers_warp_matrix(
                center=center,
                translate=offset,
                scale=scale_rate[0],
                rot=rotate,
                shear=shear)

        # warp image and keypoints
        results['img'] = self._transform(results['img'], warp_mat,
                                         (int(w), int(h)))

        if 'keypoints' in results:
            # Only transform (x, y) coordinates
            kpts = cv2.transform(results['keypoints'], warp_mat)
            if kpts.shape[-1] == 3:
                kpts = kpts[..., :2] / kpts[..., 2:3]
            results['keypoints'] = kpts

            if self.clip_border:
                results['keypoints'], results[
                    'keypoints_visible'] = keypoint_clip_border(
                        results['keypoints'], results['keypoints_visible'],
                        (w, h))

        if 'bbox' in results:
            bbox = bbox_xyxy2corner(results['bbox'])
            bbox = cv2.transform(bbox, warp_mat)
            if bbox.shape[-1] == 3:
                bbox = bbox[..., :2] / bbox[..., 2:3]
            if not self.bbox_keep_corner:
                bbox = bbox_corner2xyxy(bbox)
            if self.clip_border:
                bbox = bbox_clip_border(bbox, (w, h))
            results['bbox'] = bbox

        if 'area' in results:
            warp_mat_for_area = warp_mat
            if warp_mat.shape[0] == 2:
                aux_row = np.array([[0.0, 0.0, 1.0]], dtype=warp_mat.dtype)
                warp_mat_for_area = np.concatenate((warp_mat, aux_row))
            results['area'] *= np.linalg.det(warp_mat_for_area)

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
                 pad_val: tuple = (0, 0, 0),
                 use_udp: bool = False):
        super().__init__()

        self.input_size = input_size
        self.aug_scales = aug_scales
        self.resize_mode = resize_mode
        self.size_factor = size_factor
        self.use_udp = use_udp
        self.pad_val = pad_val

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
                img,
                warp_mat,
                padded_input_size,
                flags=cv2.INTER_LINEAR,
                borderValue=self.pad_val)

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


@TRANSFORMS.register_module()
class BottomupRandomCrop(BaseTransform):
    """Random crop the image & bboxes & masks.

    The absolute ``crop_size`` is sampled based on ``crop_type`` and
    ``image_size``, then the cropped results are generated.

    Required Keys:

        - img
        - keypoints
        - bbox (optional)
        - masks (BitmapMasks | PolygonMasks) (optional)

    Modified Keys:

        - img
        - img_shape
        - keypoints
        - keypoints_visible
        - num_keypoints
        - bbox (optional)
        - bbox_score (optional)
        - id (optional)
        - category_id (optional)
        - raw_ann_info (optional)
        - iscrowd (optional)
        - segmentation (optional)
        - masks (optional)

    Added Keys:

        - warp_mat

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            (width, height).
        crop_type (str, optional): One of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])].
            Defaults to "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Defaults to False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Defaults to False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - If the crop does not contain any gt-bbox region and
          ``allow_negative_crop`` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size: tuple,
                 crop_type: str = 'absolute',
                 allow_negative_crop: bool = False,
                 recompute_bbox: bool = False,
                 bbox_clip_border: bool = True) -> None:
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
            if crop_type == 'absolute_range':
                assert crop_size[0] <= crop_size[1]
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the warp matrix for the RandomCrop
        warp_mat = np.array([[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
                            dtype=np.float32)
        if results.get('warp_mat', None) is None:
            results['warp_mat'] = warp_mat
        else:
            results['warp_mat'] = warp_mat @ results['warp_mat']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('bbox', None) is not None:
            distances = (-offset_w, -offset_h)
            bboxes = results['bbox']
            bboxes = bboxes + np.tile(np.asarray(distances), 2)

            if self.bbox_clip_border:
                bboxes[..., 0::2] = bboxes[..., 0::2].clip(0, img_shape[1])
                bboxes[..., 1::2] = bboxes[..., 1::2].clip(0, img_shape[0])

            valid_inds = (bboxes[..., 0] < img_shape[1]) & \
                (bboxes[..., 1] < img_shape[0]) & \
                (bboxes[..., 2] > 0) & \
                (bboxes[..., 3] > 0)

            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['bbox'] = bboxes[valid_inds]
            meta_keys = [
                'bbox_score', 'id', 'category_id', 'raw_ann_info', 'iscrowd'
            ]
            for key in meta_keys:
                if results.get(key):
                    if isinstance(results[key], list):
                        results[key] = np.asarray(
                            results[key])[valid_inds].tolist()
                    else:
                        results[key] = results[key][valid_inds]

            if results.get('keypoints', None) is not None:
                keypoints = results['keypoints']
                distances = np.asarray(distances).reshape(1, 1, 2)
                keypoints = keypoints + distances
                if self.bbox_clip_border:
                    keypoints_outside_x = keypoints[:, :, 0] < 0
                    keypoints_outside_y = keypoints[:, :, 1] < 0
                    keypoints_outside_width = keypoints[:, :, 0] > img_shape[1]
                    keypoints_outside_height = keypoints[:, :,
                                                         1] > img_shape[0]

                    kpt_outside = np.logical_or.reduce(
                        (keypoints_outside_x, keypoints_outside_y,
                         keypoints_outside_width, keypoints_outside_height))

                    results['keypoints_visible'][kpt_outside] *= 0
                keypoints[:, :, 0] = keypoints[:, :, 0].clip(0, img_shape[1])
                keypoints[:, :, 1] = keypoints[:, :, 1].clip(0, img_shape[0])
                results['keypoints'] = keypoints[valid_inds]
                results['keypoints_visible'] = results['keypoints_visible'][
                    valid_inds]

            if results.get('segmentation', None) is not None:
                results['segmentation'] = results['segmentation'][
                    crop_y1:crop_y2, crop_x1:crop_x2]

            if results.get('masks', None) is not None:
                results['masks'] = results['masks'][valid_inds.nonzero(
                )[0]].crop(np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['bbox'] = results['masks'].get_bboxes(
                        type(results['bbox']))

        return results

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w

    @cache_randomness
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (Tuple[int, int]): (h, w).

        Returns:
            crop_size (Tuple[int, int]): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return min(self.crop_size[1], h), min(self.crop_size[0], w)
        elif self.crop_type == 'absolute_range':
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_w, crop_h = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        else:
            # 'relative_range'
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results


@TRANSFORMS.register_module()
class BottomupRandomChoiceResize(BaseTransform):
    """Resize images & bbox & mask from a list of multiple scales.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. Resize scale will be randomly
    selected from ``scales``.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if `scale` is a list of tuple, the target scale is sampled from the list
      uniformally.
    - if `scale` is a tuple, the target scale will be set to the tuple.

    Required Keys:

    - img
    - bbox
    - keypoints

    Modified Keys:

    - img
    - img_shape
    - bbox
    - keypoints

    Added Keys:

    - scale
    - scale_factor
    - scale_idx

    Args:
        scales (Union[list, Tuple]): Images scales for resizing.

        **resize_kwargs: Other keyword arguments for the ``resize_type``.
    """

    def __init__(
        self,
        scales: Sequence[Union[int, Tuple]],
        keep_ratio: bool = False,
        clip_object_border: bool = True,
        backend: str = 'cv2',
        **resize_kwargs,
    ) -> None:
        super().__init__()
        if isinstance(scales, list):
            self.scales = scales
        else:
            self.scales = [scales]

        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        self.backend = backend

    @cache_randomness
    def _random_select(self) -> Tuple[int, int]:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): Returns a tuple ``(scale, scale_dix)``,
            where ``scale`` is the selected image scale and
            ``scale_idx`` is the selected index in the given candidates.
        """

        scale_idx = np.random.randint(len(self.scales))
        scale = self.scales[scale_idx]
        return scale, scale_idx

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``self.scale``."""

        if self.keep_ratio:

            img, scale_factor = imrescale(
                results['img'],
                self.scale,
                interpolation='bilinear',
                return_scale=True,
                backend=self.backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = imresize(
                results['img'],
                self.scale,
                interpolation='bilinear',
                return_scale=True,
                backend=self.backend)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['scale_factor'] = (w_scale, h_scale)
        results['input_size'] = img.shape[:2]
        w, h = results['ori_shape']
        center = np.array([w / 2, h / 2], dtype=np.float32)
        scale = np.array([w, h], dtype=np.float32)
        results['input_center'] = center
        results['input_scale'] = scale

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``self.scale``."""
        if results.get('bbox', None) is not None:
            bboxes = results['bbox'] * np.tile(
                np.array(results['scale_factor']), 2)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['bbox'] = bboxes

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``self.scale``."""
        if results.get('keypoints', None) is not None:
            keypoints = results['keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['keypoints'] = keypoints

    def transform(self, results: dict) -> dict:
        """Apply resize transforms on results from a list of scales.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Resized results, 'img', 'bbox',
            'keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        target_scale, scale_idx = self._random_select()

        self.scale = target_scale
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_keypoints(results)

        results['scale_idx'] = scale_idx
        return results
