# Copyright (c) OpenMMLab. All rights reserved.
from collections import abc
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement, PixelData
from mmengine.utils import is_list_of

IndexType = Union[str, slice, int, list, torch.LongTensor,
                  torch.cuda.LongTensor, torch.BoolTensor,
                  torch.cuda.BoolTensor, np.ndarray]


class MultilevelPixelData(BaseDataElement):
    """Data structure for multi-level pixel-wise annotations or predictions.

    All data items in ``data_fields`` of ``MultilevelPixelData`` are lists
    of np.ndarray or torch.Tensor, and should meet the following requirements:

    - Have the same length, which is the number of levels
    - At each level, the data should have 3 dimensions in order of channel,
        height and weight
    - At each level, the data should have the same height and weight

    Examples:
        >>> metainfo = dict(num_keypoints=17)
        >>> sizes = [(64, 48), (128, 96), (256, 192)]
        >>> heatmaps = [np.random.rand(17, h, w) for h, w in sizes]
        >>> masks = [torch.rand(1, h, w) for h, w in sizes]
        >>> data = MultilevelPixelData(metainfo=metainfo,
        ...                            heatmaps=heatmaps,
        ...                            masks=masks)

        >>> # get data item
        >>> heatmaps = data.heatmaps  # A list of 3 numpy.ndarrays
        >>> masks = data.masks  # A list of 3 torch.Tensors

        >>> # get level
        >>> data_l0 = data[0]  # PixelData with fields 'heatmaps' and 'masks'
        >>> data.nlevel
        3

        >>> # get shape
        >>> data.shape
        ((64, 48), (128, 96), (256, 192))

        >>> # set
        >>> offset_maps = [torch.rand(2, h, w) for h, w in sizes]
        >>> data.offset_maps = offset_maps
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:
        object.__setattr__(self, '_nlevel', None)
        super().__init__(metainfo=metainfo, **kwargs)

    @property
    def nlevel(self):
        """Return the level number.

        Returns:
            Optional[int]: The level number, or ``None`` if the data has not
            been assigned.
        """
        return self._nlevel

    def __getitem__(self, item: Union[int, str, list,
                                      slice]) -> Union[PixelData, Sequence]:
        if isinstance(item, int):
            if self.nlevel is None or item >= self.nlevel:
                raise IndexError(
                    f'Lcale index {item} out of range ({self.nlevel})')
            return self.get(f'_level_{item}')

        if isinstance(item, str):
            if item not in self:
                raise KeyError(item)
            return getattr(self, item)

        # TODO: support indexing by list and slice over levels
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support index type '
            f'{type(item)}')

    def levels(self) -> List[PixelData]:
        if self.nlevel:
            return list(self[i] for i in range(self.nlevel))
        return []

    @property
    def shape(self) -> Optional[Tuple[Tuple]]:
        """Get the shape of multi-level pixel data.

        Returns:
            Optional[tuple]: A tuple of data shape at each level, or ``None``
            if the data has not been assigned.
        """
        if self.nlevel is None:
            return None

        return tuple(level.shape for level in self.levels())

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data,
                          dict), f'meta should be a `dict` but got {data}'
        for k, v in data.items():
            self.set_field(v, k, field_type='data')

    def set_field(self,
                  value: Any,
                  name: str,
                  dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
                  field_type: str = 'data') -> None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ['metainfo', 'data']
        if dtype is not None:
            assert isinstance(
                value,
                dtype), f'{value} should be a {dtype} but got {type(value)}'

        if name.startswith('_level_'):
            raise AttributeError(
                f'Cannot set {name} to be a field because the pattern '
                '<_level_{n}> is reserved for inner data field')

        if field_type == 'metainfo':
            if name in self._data_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of metainfo '
                    f'because {name} is already a data field')
            self._metainfo_fields.add(name)

        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of data '
                    f'because {name} is already a metainfo field')

            if not isinstance(value, abc.Sequence):
                raise TypeError(
                    'The value should be a sequence (of numpy.ndarray or'
                    f'torch.Tesnor), but got a {type(value)}')

            if len(value) == 0:
                raise ValueError('Setting empty value is not allowed')

            if not isinstance(value[0], (torch.Tensor, np.ndarray)):
                raise TypeError(
                    'The value should be a sequence of numpy.ndarray or'
                    f'torch.Tesnor, but got a sequence of {type(value[0])}')

            if self.nlevel is not None:
                assert len(value) == self.nlevel, (
                    f'The length of the value ({len(value)}) should match the'
                    f'number of the levels ({self.nlevel})')
            else:
                object.__setattr__(self, '_nlevel', len(value))
                for i in range(self.nlevel):
                    object.__setattr__(self, f'_level_{i}', PixelData())

            for i, v in enumerate(value):
                self[i].set_field(v, name, field_type='data')

            self._data_fields.add(name)

        object.__setattr__(self, name, value)

    def __delattr__(self, item: str):
        """delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 'private attribute, which is immutable. ')

        if item in self._metainfo_fields:
            super().__delattr__(item)
        else:
            for level in self.levels():
                level.__delattr__(item)
            self._data_fields.remove(item)

    def __getattr__(self, name):
        if name in {'_data_fields', '_metainfo_fields'
                    } or name not in self._data_fields:
            raise AttributeError(
                f'\'{self.__class__.__name__}\' object has no attribute '
                f'\'{name}\'')

        return [getattr(level, name) for level in self.levels()]

    def pop(self, *args) -> Any:
        """pop property in data and metainfo as the same as python."""
        assert len(args) < 3, '``pop`` get more than 2 arguments'
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(name)
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(name)
            return [level.pop(*args) for level in self.levels()]

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f'{args[0]} is not contained in metainfo or data')

    def _convert(self, apply_to: Type,
                 func: Callable[[Any], Any]) -> 'MultilevelPixelData':
        """Convert data items with the given function.

        Args:
            apply_to (Type): The type of data items to apply the conversion
            func (Callable): The conversion function that takes a data item
                as the input and return the converted result

        Returns:
            MultilevelPixelData: the converted data element.
        """
        new_data = self.new()
        for k, v in self.items():
            if is_list_of(v, apply_to):
                v = [func(_v) for _v in v]
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def cpu(self) -> 'MultilevelPixelData':
        """Convert all tensors to CPU in data."""
        return self._convert(apply_to=torch.Tensor, func=lambda x: x.cpu())

    def cuda(self) -> 'MultilevelPixelData':
        """Convert all tensors to GPU in data."""
        return self._convert(apply_to=torch.Tensor, func=lambda x: x.cuda())

    def detach(self) -> 'MultilevelPixelData':
        """Detach all tensors in data."""
        return self._convert(apply_to=torch.Tensor, func=lambda x: x.detach())

    def numpy(self) -> 'MultilevelPixelData':
        """Convert all tensor to np.narray in data."""
        return self._convert(
            apply_to=torch.Tensor, func=lambda x: x.detach().cpu().numpy())

    def to_tensor(self) -> 'MultilevelPixelData':
        """Convert all tensor to np.narray in data."""
        return self._convert(
            apply_to=np.ndarray, func=lambda x: torch.from_numpy(x))

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'MultilevelPixelData':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v[0], 'to'):
                v = [v_.to(*args, **kwargs) for v_ in v]
                data = {k: v}
                new_data.set_data(data)
        return new_data
