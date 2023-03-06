# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes
from torch import Tensor

DeviceType = Union[str, torch.device]
T = TypeVar('T')
IndexType = Union[slice, int, list, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor, np.ndarray]


class BBoxKeypoints(HorizontalBoxes):
    """The BBoxKeypoints class is a combination of bounding boxes and keypoints
    representation. The box format used in BBoxKeypoints is the same as
    HorizontalBoxes.

    Args:
        data (Tensor or np.ndarray): The box data with shape of
            (N, 4).
        keypoints (Tensor or np.ndarray): The keypoint data with shape of
            (N, K, 2).
        keypoints_visible (Tensor or np.ndarray): The visibility of keypoints
            with shape of (N, K).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
        mode (str, Optional): the mode of boxes. If it is 'cxcywh', the
            `data` will be converted to 'xyxy' mode. Defaults to None.
        flip_indices (list, Optional): The indices of keypoints when the
            images is flipped. Defaults to None.

    Notes:
        N: the number of instances.
        K: the number of keypoints.
    """

    def __init__(self,
                 data: Union[Tensor, np.ndarray],
                 keypoints: Union[Tensor, np.ndarray],
                 keypoints_visible: Union[Tensor, np.ndarray],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[DeviceType] = None,
                 clone: bool = True,
                 in_mode: Optional[str] = None,
                 flip_indices: Optional[List] = None) -> None:

        super().__init__(
            data=data,
            dtype=dtype,
            device=device,
            clone=clone,
            in_mode=in_mode)

        assert len(data) == len(keypoints)
        assert len(data) == len(keypoints_visible)

        assert keypoints.ndim == 3
        assert keypoints_visible.ndim == 2

        keypoints = torch.as_tensor(keypoints)
        keypoints_visible = torch.as_tensor(keypoints_visible)

        if device is not None:
            keypoints = keypoints.to(device=device)
            keypoints_visible = keypoints_visible.to(device=device)

        if clone:
            keypoints = keypoints.clone()
            keypoints_visible = keypoints_visible.clone()

        self.keypoints = keypoints
        self.keypoints_visible = keypoints_visible
        self.flip_indices = flip_indices

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes & kpts horizontally in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        assert direction == 'horizontal'
        super().flip_(img_shape, direction)
        self.keypoints[..., 0] = img_shape[1] - self.keypoints[..., 0]
        self.keypoints = self.keypoints[:, self.flip_indices]
        self.keypoints_visible = self.keypoints_visible[:, self.flip_indices]

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes and keypoints in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        boxes = self.tensor
        assert len(distances) == 2
        self.tensor = boxes + boxes.new_tensor(distances).repeat(2)
        distances = self.keypoints.new_tensor(distances).reshape(1, 1, 2)
        self.keypoints = self.keypoints + distances

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes & keypoints w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        boxes = self.tensor
        assert len(scale_factor) == 2

        self.tensor = boxes * boxes.new_tensor(scale_factor).repeat(2)
        scale_factor = self.keypoints.new_tensor(scale_factor).reshape(1, 1, 2)
        self.keypoints = self.keypoints * scale_factor

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip bounding boxes and set invisible keypoints outside the image
        boundary in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        boxes = self.tensor
        boxes[..., 0::2] = boxes[..., 0::2].clamp(0, img_shape[1])
        boxes[..., 1::2] = boxes[..., 1::2].clamp(0, img_shape[0])

        kpt_outside = torch.logical_or(
            torch.logical_or(self.keypoints[..., 0] < 0,
                             self.keypoints[..., 1] < 0),
            torch.logical_or(self.keypoints[..., 0] > img_shape[1],
                             self.keypoints[..., 1] > img_shape[0]))
        self.keypoints_visible[kpt_outside] *= 0

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometrically transform bounding boxes and keypoints in-place using
        a homography matrix.

        Args:
            homography_matrix (Tensor or np.ndarray): A 3x3 tensor or ndarray
                representing the homography matrix for the transformation.
        """
        boxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)

        # Convert boxes to corners in homogeneous coordinates
        corners = self.hbox2corner(boxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)

        # Convert keypoints to homogeneous coordinates
        keypoints = torch.cat([
            self.keypoints,
            self.keypoints.new_ones(*self.keypoints.shape[:-1], 1)
        ],
                              dim=-1)

        # Transpose corners and keypoints for matrix multiplication
        corners_T = torch.transpose(corners, -1, -2)
        keypoints_T = torch.transpose(keypoints, -1, 0).contiguous().flatten(1)

        # Apply homography matrix to corners and keypoints
        corners_T = torch.matmul(homography_matrix, corners_T)
        keypoints_T = torch.matmul(homography_matrix, keypoints_T)

        # Transpose back to original shape
        corners = torch.transpose(corners_T, -1, -2)
        keypoints_T = keypoints_T.reshape(3, self.keypoints.shape[1], -1)
        keypoints = torch.transpose(keypoints_T, -1, 0).contiguous()

        # Convert corners and keypoints back to non-homogeneous coordinates
        corners = corners[..., :2] / corners[..., 2:3]
        keypoints = keypoints[..., :2] / keypoints[..., 2:3]

        # Convert corners back to bounding boxes and update object attributes
        self.tensor = self.corner2hbox(corners)
        self.keypoints = keypoints

    @classmethod
    def cat(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        """Cancatenates an instance list into one single instance. Similar to
        ``torch.cat``.

        Args:
            box_list (Sequence[T]): A sequence of instances.
            dim (int): The dimension over which the box and keypoint are
                concatenated. Defaults to 0.

        Returns:
            T: Concatenated instance.
        """
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError('box_list should not be a empty list.')

        assert dim == 0
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = torch.cat([boxes.tensor for boxes in box_list], dim=dim)
        th_kpt_list = torch.cat([boxes.keypoints for boxes in box_list],
                                dim=dim)
        th_kpt_vis_list = torch.cat(
            [boxes.keypoints_visible for boxes in box_list], dim=dim)
        flip_indices = box_list[0].flip_indices
        return cls(
            th_box_list,
            th_kpt_list,
            th_kpt_vis_list,
            clone=False,
            flip_indices=flip_indices)

    def __getitem__(self: T, index: IndexType) -> T:
        """Rewrite getitem to protect the last dimension shape."""
        boxes = self.tensor
        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.device)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < boxes.dim()
        elif isinstance(index, tuple):
            assert len(index) < boxes.dim()
            # `Ellipsis`(...) is commonly used in index like [None, ...].
            # When `Ellipsis` is in index, it must be the last item.
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        boxes = boxes[index]
        keypoints = self.keypoints[index]
        keypoints_visible = self.keypoints_visible[index]
        if boxes.dim() == 1:
            boxes = boxes.reshape(1, -1)
            keypoints = keypoints.reshape(1, -1, 2)
            keypoints_visible = keypoints_visible.reshape(1, -1)
        return type(self)(
            boxes,
            keypoints,
            keypoints_visible,
            flip_indices=self.flip_indices,
            clone=False)

    @property
    def num_keypoints(self) -> Tensor:
        """Compute the number of visible keypoints for each object."""
        return self.keypoints_visible.sum(dim=1).int()

    def __deepcopy__(self, memo):
        """Only clone the tensors when applying deepcopy."""
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other
        other.tensor = self.tensor.clone()
        other.keypoints = self.keypoints.clone()
        other.keypoints_visible = self.keypoints_visible.clone()
        other.flip_indices = deepcopy(self.flip_indices)
        return other

    def clone(self: T) -> T:
        """Reload ``clone`` for tensors."""
        return type(self)(
            self.tensor,
            self.keypoints,
            self.keypoints_visible,
            flip_indices=self.flip_indices,
            clone=True)

    def to(self: T, *args, **kwargs) -> T:
        """Reload ``to`` for tensors."""
        return type(self)(
            self.tensor.to(*args, **kwargs),
            self.keypoints.to(*args, **kwargs),
            self.keypoints_visible.to(*args, **kwargs),
            flip_indices=self.flip_indices,
            clone=False)
