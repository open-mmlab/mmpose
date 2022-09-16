# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_seq_of

from mmpose.registry import TRANSFORMS
from mmpose.structures import MultilevelPixelData, PoseDataSample


def image_to_tensor(img: Union[np.ndarray,
                               Sequence[np.ndarray]]) -> torch.torch.Tensor:
    """Translate image or sequence of images to tensor. Multiple image tensors
    will be stacked.

    Args:
        value (np.ndarray | Sequence[np.ndarray]): The original image or
            image sequence

    Returns:
        torch.Tensor: The output tensor.
    """

    if isinstance(img, np.ndarray):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        tensor = to_tensor(img)
    else:
        assert is_seq_of(img, np.ndarray)
        tensor = torch.stack([image_to_tensor(_img) for _img in img])

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
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``input_size``: the input size to the network

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

        - ``flip_indices``: the indices of each keypoint's symmetric keypoint

    Args:
        meta_keys (Sequence[str], optional): Meta keys which will be stored in
            :obj: `PoseDataSample` as meta info. Defaults to ``('id',
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'input_size',
            'flip', 'flip_direction', 'flip_indices)``
    """

    # items in `instance_mapping_table` will be directly packed into
    # PoseDataSample without converting to Tensor
    instance_mapping_table = {
        'bbox': 'bboxes',
        'head_size': 'head_size',
        'bbox_center': 'bbox_centers',
        'bbox_scale': 'bbox_scales',
        'bbox_score': 'bbox_scores',
        'keypoints': 'keypoints',
        'keypoints_visible': 'keypoints_visible'
    }

    label_mapping_table = {
        'keypoint_labels': 'keypoint_labels',
        'keypoint_x_labels': 'keypoint_x_labels',
        'keypoint_y_labels': 'keypoint_y_labels',
        'keypoint_weights': 'keypoint_weights'
    }

    field_mapping_table = {
        'heatmaps': 'heatmaps',
    }

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'ori_shape',
                            'img_shape', 'input_size', 'flip',
                            'flip_direction', 'flip_indices'),
                 pack_transformed=False):
        self.meta_keys = meta_keys
        self.pack_transformed = pack_transformed

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_samples' (obj:`PoseDataSample`): The annotation info of the
                sample.
        """
        # Pack image(s)
        if 'img' in results:
            img = results['img']
            img_tensor = image_to_tensor(img)

        data_sample = PoseDataSample()

        # pack instance data
        gt_instances = InstanceData()
        for key, packed_key in self.instance_mapping_table.items():
            if key in results:
                gt_instances.set_field(results[key], packed_key)

        # pack `transformed_keypoints` for visualizing data transform
        # and augmentation results
        if self.pack_transformed and 'transformed_keypoints' in results:
            gt_instances.set_field(results['transformed_keypoints'],
                                   'transformed_keypoints')

        data_sample.gt_instances = gt_instances

        # pack instance labels
        gt_instance_labels = InstanceData()
        for key, packed_key in self.label_mapping_table.items():
            if key in results:
                gt_instance_labels.set_field(results[key], packed_key)
        data_sample.gt_instance_labels = gt_instance_labels.to_tensor()

        # pack fields
        gt_fields = None
        for key, packed_key in self.field_mapping_table.items():
            if key in results:
                if isinstance(results[key], list):
                    if gt_fields is None:
                        gt_fields = MultilevelPixelData()
                    else:
                        assert isinstance(
                            gt_fields, MultilevelPixelData
                        ), 'Got mixed single-level and multi-level pixel data.'
                else:
                    if gt_fields is None:
                        gt_fields = PixelData()
                    else:
                        assert isinstance(
                            gt_fields, PixelData
                        ), 'Got mixed single-level and multi-level pixel data.'

                gt_fields.set_field(results[key], packed_key)

        if gt_fields:
            data_sample.gt_fields = gt_fields.to_tensor()

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)

        packed_results = dict()
        packed_results['inputs'] = img_tensor
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
