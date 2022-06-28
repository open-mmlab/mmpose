# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_list_of, is_seq_of

from mmpose.core.bbox import get_udp_warp_matrix, get_warp_matrix
from mmpose.core.keypoint import (generate_megvii_heatmap,
                                  generate_msra_heatmap, generate_udp_heatmap)
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TopDownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints
        - keypoints_visible

    Modified Keys:

        - img
        - keypoints
        - bbox_scale

    Added Keys:

        - input_size

    Args:
        input_size (tuple): The input image size of the model in [w, h]. The
            bbox region will be cropped and resize to `input_size`
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

        w, h = np.hsplit(bbox_scale, [1])
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

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
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

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str


@TRANSFORMS.register_module()
class TopDownGenerateHeatmap(BaseTransform):
    r"""Generate the target heatmap of the keypoints.

    Required Keys:

        - keypoints
        - keypoints_visible
        - keypoint_weights

    Added Keys:

        - gt_heatmap
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

        assert is_seq_of(heatmap_size, int) and len(heatmap_size) == 2, (
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
        self.encoder, self.encoder_kwargs = self._get_encoder()

    def _get_encoder(self):
        """Get heatmap generation function and arguments.

        Returns:
            tuple:
            - encoder [callable]: The heatmap generation function
            - encoder_kwargs [dict | list[dict]]: The keyword arguments of
                ``encoder``. A list means to generate multi-level
                heatmaps where each element is the keyword arguments of
                one level
        """

        if self.encoding == 'msra':
            encoder = generate_msra_heatmap

            if isinstance(self.sigma, (list, tuple)):
                encoder_kwargs = [{
                    'sigma': sigma,
                    'unbiased': self.unbiased
                } for sigma in self.sigma]
            else:
                encoder_kwargs = {
                    'sigma': self.sigma,
                    'unbiased': self.unbiased
                }

        elif self.encoding == 'megvii':
            encoder = generate_megvii_heatmap

            if is_list_of(self.kernel_size, (list, tuple)):
                encoder_kwargs = [{
                    'kernel_size': kernel_size
                } for kernel_size in self.kernel_size]
            else:
                encoder_kwargs = {'kernel_size': self.kernel_size}

        elif self.encoding == 'udp':
            encoder = generate_udp_heatmap

            if self.udp_combined_map:
                factor = self.udp_radius_factor
            else:
                factor = self.sigma

            if isinstance(factor, (list, tuple)):
                encoder_kwargs = [{
                    'factor': _factor,
                    'combined_map': self.udp_combined_map
                } for _factor in factor]
            else:
                encoder_kwargs = {
                    'factor': factor,
                    'combined_map': self.udp_combined_map
                }

        else:
            raise ValueError(f'Invalid encoding type {self.encoding}')

        return encoder, encoder_kwargs

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

        encoder_kwargs = deepcopy(self.encoder_kwargs)

        if isinstance(encoder_kwargs, list):
            # multi-level heatmap
            heatmaps = []
            keypoint_weights = []

            for _kwargs in encoder_kwargs:
                _heatmap, _keypoint_weight = self.encoder(
                    keypoints=keypoints,
                    keypoints_visible=keypoints_visible,
                    image_size=results['input_size'],
                    heatmap_size=self.heatmap_size,
                    **_kwargs)

                heatmaps.append(_heatmap)
                keypoint_weights.append(_keypoint_weight)

            heatmap = np.stack(heatmaps)
            keypoint_weight = np.stack(keypoint_weights)

        else:
            # single-level heatmap
            heatmap, keypoint_weight = self.encoder(
                keypoints=keypoints,
                keypoints_visible=keypoints_visible,
                image_size=results['input_size'],
                heatmap_size=self.heatmap_size,
                **encoder_kwargs)

        # multiply meta keypoint weight
        if self.use_meta_keypoint_weight:
            keypoint_weight *= results['keypoint_weights'][:, None]

        results['gt_heatmap'] = heatmap
        results['target_weight'] = keypoint_weight

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(heatmap_size={self.heatmap_size}, '
        repr_str += f'encoding="{self.encoding}", '
        if self.encoding == 'msra':
            repr_str += f'sigma={self.sigma}, '
            repr_str += f'unbiased={self.unbiased}, '
        elif self.encoding == 'megvii':
            repr_str += f'kernel_size={self.kernel_size}, '
        elif self.encoding == 'udp':
            repr_str += f'combined_map={self.udp_combined_map}, '
            if self.udp_combined_map:
                repr_str += f'radius_factor={self.udp_radius_factor}, '
            else:
                repr_str += f'sigma={self.sigma}, '
        repr_str += ('use_meta_keypoint_weight='
                     f'{self.use_meta_keypoint_weight})')
        return repr_str


@TRANSFORMS.register_module()
class TopDownGenerateRegressionLabel(BaseTransform):
    """Generate the target regression label of the keypoints.

    Required Keys:

        - keypoints
        - keypoints_visible
        - image_size

    Added Keys:

        - gt_reg_label
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
        valid = ((keypoints >= 0) & (keypoints <= [w - 1, h - 1]) &
                 (keypoints_visible > 0.5)).all(
                     axis=1, keepdims=True)

        reg_label = keypoints / [w, h]
        target_weight = np.where(valid, 1., 0.).astype(np.float32)

        # multiply meta keypoint weight
        if self.use_meta_keypoint_weight:
            target_weight *= results['keypoint_weights'][:, None]

        results['gt_reg_label'] = reg_label
        results['target_weight'] = target_weight

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += ('(use_meta_keypoint_weight='
                     f'{self.use_meta_keypoint_weight})')
        return repr_str
