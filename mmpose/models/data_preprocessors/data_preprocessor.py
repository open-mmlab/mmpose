# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from mmengine.model import ImgDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.utils import is_seq_of
from PIL import Image

from mmpose.registry import MODELS


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks.

    Comparing with the :class:`ImgDataPreprocessor`,

    1. It will additionally append batch_input_shape
    to data_samples considering the DETR-based pose estimation tasks.

    2. Add a 'pillow backend' pipeline based normalize operation, convert
    np.array to PIL.Image, and normalize it through torchvision.

    3. Support image augmentation transforms on batched data.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Apply batch augmentation transforms.

    Args:
        mean (sequence of float, optional): The pixel mean of R, G, B
            channels. Defaults to None.
        std (sequence of float, optional): The pixel standard deviation
            of R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to BGR.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments: (list of dict, optional): Configs of augmentation
            transforms on batched data. Defaults to None.
        normalize_bakend (str): choose the normalize backend
            in ['cv2', 'pillow']
    """

    def __init__(self,
                 mean: Sequence[float] = None,
                 std: Sequence[float] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None,
                 normalize_bakend: str = 'cv2'):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)

        assert normalize_bakend in ('cv2', 'pillow'), f'the argument ' \
            f'`normalize_bakend` mush be either \'cv2\' or \'pillow\', ' \
            f'but got \'{normalize_bakend}\'.'

        self.normalize_bakend = normalize_bakend

        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = self.preprocess(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        # update metainfo since the image shape might change
        batch_input_shape = tuple(inputs[0].size()[-2:])
        for data_sample, pad_shape in zip(data_samples, batch_pad_shape):

            aux_metainfo = {
                'batch_input_shape': batch_input_shape,
                'pad_shape': pad_shape
            }

            if 'input_size' not in data_sample.metainfo:
                aux_metainfo['input_size'] = data_sample.img_shape

            if 'input_center' not in data_sample.metainfo \
                    or 'input_scale' not in data_sample.metainfo:

                w, h = data_sample.ori_shape
                center = np.array([w / 2, h / 2], dtype=np.float32)
                scale = np.array([w, h], dtype=np.float32)
                aux_metainfo['input_center'] = center
                aux_metainfo['input_scale'] = scale

            data_sample.set_metainfo(aux_metainfo)

        # apply batch augmentations
        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape

    def preprocess(self,
                   data: dict,
                   training: bool = False) -> Union[dict, list]:

        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]

                if self.normalize_bakend == 'cv2':
                    _batch_input = _batch_input.float()
                elif self.normalize_bakend == 'pillow':
                    _batch_input_array = _batch_input.detach().cpu().numpy(
                    ).transpose(1, 2, 0)
                    assert _batch_input_array.dtype == np.uint8, \
                        'Pillow backend only support uint8 type'
                    pil_image = Image.fromarray(_batch_input_array)
                    _batch_input = tvF.to_tensor(pil_image).to(
                        _batch_input.device)

                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    if self.normalize_bakend == 'cv2':
                        _batch_input = (_batch_input - self.mean) / self.std
                    elif self.normalize_bakend == 'pillow':
                        _batch_input = tvF.normalize(
                            _batch_input, mean=self.mean, std=self.std)
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            if self.normalize_bakend == 'cv2':
                _batch_input = _batch_input.float()
            elif self.normalize_bakend == 'pillow':
                _batch_inputs_array = _batch_inputs.detach().cpu().numpy(
                ).transpose(0, 2, 3, 1)
                assert _batch_inputs.dtype == np.uint8, \
                    'Pillow backend only support uint8 type'
                pil_image = Image.fromarray(_batch_inputs_array)
                _batch_inputs = tvF.to_tensor(pil_image).to(
                    _batch_inputs.device)

            if self._enable_normalize:
                if self.normalize_bakend == 'cv2':
                    _batch_inputs = (_batch_inputs - self.mean) / self.std
                elif self.normalize_bakend == 'pillow':
                    _batch_inputs = tvF.normalize(
                        _batch_inputs,
                        mean=(self.mean / 255).tolist(),
                        std=(self.std / 255).tolist())

            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data
