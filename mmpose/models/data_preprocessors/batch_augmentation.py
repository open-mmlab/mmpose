# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.structures import PixelData
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.structures import PoseDataSample


@MODELS.register_module()
class BatchSyncRandomResize(nn.Module):
    """Batch random resize which synchronizes the random size across ranks.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 10,
                 size_divisor: int = 32) -> None:
        super().__init__()
        self.rank, self.world_size = get_dist_info()
        self._input_size = None
        self._random_size_range = (round(random_size_range[0] / size_divisor),
                                   round(random_size_range[1] / size_divisor))
        self._interval = interval
        self._size_divisor = size_divisor

    def forward(self, inputs: Tensor, data_samples: List[PoseDataSample]
                ) -> Tuple[Tensor, List[PoseDataSample]]:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for data_sample in data_samples:
                img_shape = (int(data_sample.img_shape[0] * scale_y),
                             int(data_sample.img_shape[1] * scale_x))
                pad_shape = (int(data_sample.pad_shape[0] * scale_y),
                             int(data_sample.pad_shape[1] * scale_x))
                data_sample.set_metainfo({
                    'img_shape': img_shape,
                    'pad_shape': pad_shape,
                    'batch_input_shape': self._input_size
                })

                if 'gt_instance_labels' not in data_sample:
                    continue

                if 'bboxes' in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.bboxes[..., 0::2] *= scale_x
                    data_sample.gt_instance_labels.bboxes[..., 1::2] *= scale_y

                if 'keypoints' in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.keypoints[..., 0] *= scale_x
                    data_sample.gt_instance_labels.keypoints[..., 1] *= scale_y

                if 'areas' in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.areas *= scale_x * scale_y

                if 'gt_fields' in data_sample \
                        and 'heatmap_mask' in data_sample.gt_fields:

                    mask = data_sample.gt_fields.heatmap_mask.unsqueeze(0)
                    gt_fields = PixelData()
                    gt_fields.set_field(
                        F.interpolate(
                            mask.float(),
                            size=self._input_size,
                            mode='bilinear',
                            align_corners=False).squeeze(0), 'heatmap_mask')

                    data_sample.gt_fields = gt_fields

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=inputs.device)
        return inputs, data_samples

    def _get_random_size(self, aspect_ratio: float,
                         device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.LongTensor(2).to(device)
        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_divisor * size,
                    self._size_divisor * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
