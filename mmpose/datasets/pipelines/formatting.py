# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.data import InstanceData, PixelData

from mmpose.core import PoseDataSample
from mmpose.registry import TRANSFORMS


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Trans image to tensor.

    Args:
        img (np.ndarray): The original image.

    Returns:
        torch.Tensor: The output tensor.
    """

    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    tensor = to_tensor(img)

    return tensor


def images_to_tensor(
    value: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, np.ndarray]]
) -> torch.Tensor:
    """Translate image or sequence of frames to tensor.

    Args:
        value (np.ndarray | List[np.ndarray] | Tuple[np.ndarray, np.ndarray]):
            The original image or list of frames.

    Returns:
        torch.Tensor: The output tensor.
    """

    if isinstance(value, (List, Tuple)):
        frames = [image_to_tensor(v) for v in value]
        # stack sequence of frames
        # List[Tensor[c, h, w]] -> Tensor[len_seq, c, h, w]
        tensor = torch.stack(frames, dim=0)
    elif isinstance(value, np.ndarray):
        tensor = image_to_tensor(value)
    else:
        # Maybe the data has been converted to Tensor.
        tensor = to_tensor(value)

    return tensor


@TRANSFORMS.register_module()
class PackPoseInputs(BaseTransform):
    """Pack the inputs data for pose estimation.

    The ``img_meta`` item is always populated. The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default it includes:

        - ``id``: id of the data sample

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``input_size``: the input size to the network

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Defaults to ``('id', 'img_id', 'img_path', 'ori_shape',
            'img_shape', 'input_size', 'flip', 'flip_direction')``
    """

    mapping_table = {
        'bbox': 'bboxes',
        'bbox_center': 'bbox_centers',
        'bbox_scale': 'bbox_scales',
        'bbox_score': 'bbox_scores',
        'keypoints': 'keypoints',
        'keypoints_visible': 'keypoints_visible',
        'reg_label': 'reg_labels',
        'keypoint_weights': 'keypoint_weights'
    }

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'ori_shape',
                            'img_shape', 'input_size', 'flip',
                            'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`PoseDataSample`): The annotation info of the
                sample.
        """
        if 'img' in results:
            img = results['img']
            img_tensor = images_to_tensor(img)

        data_sample = PoseDataSample()
        gt_instances = InstanceData()
        gt_fields = PixelData()

        for key, packed_key in self.mapping_table.items():
            if key in results:
                gt_instances[packed_key] = results[key]

        if 'heatmaps' in results:
            gt_fields.heatmaps = results['heatmaps']

        data_sample.gt_instances = gt_instances
        data_sample.gt_fields = gt_fields

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)

        packed_results = dict()
        packed_results['inputs'] = img_tensor
        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
