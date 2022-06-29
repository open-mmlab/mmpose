# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.data import InstanceData, PixelData

from mmpose.core import PoseDataSample
from mmpose.registry import TRANSFORMS


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

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Defaults to ``('id', 'img_id', 'img_path', 'ori_shape',
            'img_shape', 'scale_factor', 'flip', 'flip_direction')``
    """

    INSTANCE_KEYS = [
        'bbox', 'bbox_center', 'bbox_scale', 'bbox_rotation', 'bbox_score',
        'keypoints', 'keypoints_visible', 'gt_reg_label', 'target_weight',
        'mask_invalid_rle'
    ]

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'ori_shape',
                            'img_shape', 'scale_factor', 'flip',
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
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = to_tensor(img)

        data_sample = PoseDataSample()
        gt_instances = InstanceData()
        gt_fields = PixelData()

        for key in self.INSTANCE_KEYS:
            if key in results:
                gt_instances[key] = results[key]

        if 'gt_heatmap' in results:
            gt_fields.heatmaps = results['gt_heatmap']

        data_sample.gt_instances = gt_instances
        data_sample.gt_fields = gt_fields

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)

        packed_results = dict()
        packed_results['inputs'] = img
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
