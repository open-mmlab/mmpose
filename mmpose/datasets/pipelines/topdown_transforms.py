# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

from mmpose.core.bbox import get_udp_warp_matrix, get_warp_matrix
from mmpose.core.utils.typing import MultiConfig
from mmpose.registry import KEYPOINT_CODECS, TRANSFORMS


@TRANSFORMS.register_module()
class TopdownAffine(BaseTransform):
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
        """The transform function of :class:`TopdownAffine`.

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
class TopdownGenerateHeatmap(BaseTransform):
    """Encode keypoints into heatmaps.

    Required Keys:

        - keypoints
        - keypoints_visible
        - dataset_keypoint_weights

    Added Keys:
        - heatmaps
        - keypoint_weights

    Args:
        encoder (dict | list[dict])
    """

    def __init__(self,
                 encoder: MultiConfig,
                 use_dataset_keypoint_weights: bool = False) -> None:
        super().__init__()
        self.encoder_cfg = deepcopy(encoder)
        self.use_dataset_keypoint_weights = use_dataset_keypoint_weights

        if isinstance(encoder, list):
            self.encoder = [KEYPOINT_CODECS.build(cfg) for cfg in encoder]
        else:
            self.encoder = KEYPOINT_CODECS.build(encoder)

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        if isinstance(self.encoder, list):
            # multi-level heatmaps
            heatmaps = []
            keypoint_weights = []

            for encoder in self.encoder:
                _heatmaps, _keypoints_weights = encoder.encode(
                    keypoints=results['keypoints'],
                    keypoints_visible=results['keypoints_visible'])

                heatmaps.append(_heatmaps)
                keypoint_weights.append(_keypoints_weights)

            results['heatmaps'] = np.stack(heatmaps)
            results['keypoint_weights'] = np.stack(keypoint_weights)
        else:
            # single-level heatmaps
            heatmaps, keypoint_weights = self.encoder.encode(
                keypoints=results['keypoints'],
                keypoints_visible=results['keypoints_visible'])

            results['heatmaps'] = heatmaps
            results['keypoint_weights'] = keypoint_weights

        # multiply meta keypoint weight
        if self.use_dataset_keypoint_weights:
            results['keypoint_weights'] *= results['dataset_keypoint_weights']

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(encoder={str(self.encoder_cfg)}, ')
        repr_str += ('use_dataset_keypoint_weights='
                     f'{self.use_dataset_keypoint_weights})')
        return repr_str


@TRANSFORMS.register_module()
class TopdownGenerateRegressionLabel(BaseTransform):
    """Generate the target regression label of the keypoints.

    Required Keys:

        - keypoints
        - keypoints_visible
        - image_size
        - dataset_keypoint_weights

    Added Keys:

        - reg_label
        - keypoint_weights

    Args:
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
    """

    def __init__(self, use_dataset_keypoint_weights: bool = False) -> None:
        super().__init__()
        self.use_dataset_keypoint_weights = use_dataset_keypoint_weights

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function of :class:`TopdownGenerateRegressionLabel`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        keypoints = results['keypoints']
        keypoints_visible = results['keypoints_visible']

        w, h = results['input_size']
        valid = ((keypoints >= 0) &
                 (keypoints <= [w - 1, h - 1])).all(axis=-1) & (
                     keypoints_visible > 0.5)

        reg_label = keypoints / [w, h]
        keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

        # multiply meta keypoint weight
        if self.use_dataset_keypoint_weights:
            keypoint_weights *= results['dataset_keypoint_weights']

        results['reg_label'] = reg_label
        results['keypoint_weights'] = keypoint_weights

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += ('(use_dataset_keypoint_weights='
                     f'{self.use_dataset_keypoint_weights})')
        return repr_str
