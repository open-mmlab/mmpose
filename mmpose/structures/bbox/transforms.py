# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import cv2
import numpy as np


def bbox_xyxy2xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh


def bbox_xywh2xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]

    return bbox_xyxy


def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def bbox_xywh2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (x, y, h, w)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """

    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x, y, w, h = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x + w * 0.5, y + h * 0.5])
    scale = np.hstack([w, h]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def bbox_cs2xyxy(center: np.ndarray,
                 scale: np.ndarray,
                 padding: float = 1.) -> np.ndarray:
    """Transform the bbox format from (center, scale) to (x1,y1,x2,y2).

    Args:
        center (ndarray): BBox center (x, y) in shape (2,) or (n, 2)
        scale (ndarray): BBox scale (w, h) in shape (2,) or (n, 2)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        ndarray[float32]: BBox (x1, y1, x2, y2) in shape (4, ) or (n, 4)
    """

    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, xy + wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_cs2xywh(center: np.ndarray,
                 scale: np.ndarray,
                 padding: float = 1.) -> np.ndarray:
    """Transform the bbox format from (center, scale) to (x,y,w,h).

    Args:
        center (ndarray): BBox center (x, y) in shape (2,) or (n, 2)
        scale (ndarray): BBox scale (w, h) in shape (2,) or (n, 2)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        ndarray[float32]: BBox (x, y, w, h) in shape (4, ) or (n, 4)
    """

    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_xyxy2corner(bbox: np.ndarray):
    """Convert bounding boxes from xyxy format to corner format.

    Given a numpy array containing bounding boxes in the format
    (xmin, ymin, xmax, ymax), this function converts the bounding
    boxes to the corner format, where each box is represented by four
    corner points (top-left, top-right, bottom-right, bottom-left).

    Args:
        bbox (numpy.ndarray): Input array of shape (N, 4) representing
            N bounding boxes.

    Returns:
        numpy.ndarray: An array of shape (N, 4, 2) containing the corner
            points of the bounding boxes.

    Example:
        bbox = np.array([[0, 0, 100, 50], [10, 20, 200, 150]])
        corners = bbox_xyxy2corner(bbox)
    """
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None]

    bbox = np.tile(bbox, 2).reshape(-1, 4, 2)
    bbox[:, 1:3, 0] = bbox[:, 0:2, 0]

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_corner2xyxy(bbox: np.ndarray):
    """Convert bounding boxes from corner format to xyxy format.

    Given a numpy array containing bounding boxes in the corner
    format (four corner points for each box), this function converts
    the bounding boxes to the (xmin, ymin, xmax, ymax) format.

    Args:
        bbox (numpy.ndarray): Input array of shape (N, 4, 2) representing
            N bounding boxes.

    Returns:
        numpy.ndarray: An array of shape (N, 4) containing the bounding
            boxes in xyxy format.

    Example:
        corners = np.array([[[0, 0], [100, 0], [100, 50], [0, 50]],
            [[10, 20], [200, 20], [200, 150], [10, 150]]])
        bbox = bbox_corner2xyxy(corners)
    """
    if bbox.shape[-1] == 8:
        bbox = bbox.reshape(*bbox.shape[:-1], 4, 2)

    dim = bbox.ndim
    if dim == 2:
        bbox = bbox[None]

    bbox = np.concatenate((bbox.min(axis=1), bbox.max(axis=1)), axis=1)

    if dim == 2:
        bbox = bbox[0]

    return bbox


def bbox_clip_border(bbox: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Clip bounding box coordinates to fit within a specified shape.

    Args:
        bbox (np.ndarray): Bounding box coordinates of shape (..., 4)
            or (..., 2).
        shape (Tuple[int, int]): Shape of the image to which bounding
            boxes are being clipped in the format of (w, h)

    Returns:
        np.ndarray: Clipped bounding box coordinates.

    Example:
        >>> bbox = np.array([[10, 20, 30, 40], [40, 50, 80, 90]])
        >>> shape = (50, 50)  # Example image shape
        >>> clipped_bbox = bbox_clip_border(bbox, shape)
    """
    width, height = shape[:2]

    if bbox.shape[-1] == 2:
        bbox[..., 0] = np.clip(bbox[..., 0], a_min=0, a_max=width)
        bbox[..., 1] = np.clip(bbox[..., 1], a_min=0, a_max=height)
    else:
        bbox[..., ::2] = np.clip(bbox[..., ::2], a_min=0, a_max=width)
        bbox[..., 1::2] = np.clip(bbox[..., 1::2], a_min=0, a_max=height)

    return bbox


def flip_bbox(bbox: np.ndarray,
              image_size: Tuple[int, int],
              bbox_format: str = 'xywh',
              direction: str = 'horizontal') -> np.ndarray:
    """Flip the bbox in the given direction.

    Args:
        bbox (np.ndarray): The bounding boxes. The shape should be (..., 4)
            if ``bbox_format`` is ``'xyxy'`` or ``'xywh'``, and (..., 2) if
            ``bbox_format`` is ``'center'``
        image_size (tuple): The image shape in [w, h]
        bbox_format (str): The bbox format. Options are ``'xywh'``, ``'xyxy'``
            and ``'center'``.
        direction (str): The flip direction. Options are ``'horizontal'``,
            ``'vertical'`` and ``'diagonal'``. Defaults to ``'horizontal'``

    Returns:
        np.ndarray: The flipped bounding boxes.
    """
    direction_options = {'horizontal', 'vertical', 'diagonal'}
    assert direction in direction_options, (
        f'Invalid flipping direction "{direction}". '
        f'Options are {direction_options}')

    format_options = {'xywh', 'xyxy', 'center'}
    assert bbox_format in format_options, (
        f'Invalid bbox format "{bbox_format}". '
        f'Options are {format_options}')

    bbox_flipped = bbox.copy()
    w, h = image_size

    # TODO: consider using "integer corner" coordinate system
    if direction == 'horizontal':
        if bbox_format == 'xywh' or bbox_format == 'center':
            bbox_flipped[..., 0] = w - bbox[..., 0] - 1
        elif bbox_format == 'xyxy':
            bbox_flipped[..., ::2] = w - bbox[..., -2::-2] - 1
    elif direction == 'vertical':
        if bbox_format == 'xywh' or bbox_format == 'center':
            bbox_flipped[..., 1] = h - bbox[..., 1] - 1
        elif bbox_format == 'xyxy':
            bbox_flipped[..., 1::2] = h - bbox[..., ::-2] - 1
    elif direction == 'diagonal':
        if bbox_format == 'xywh' or bbox_format == 'center':
            bbox_flipped[..., :2] = [w, h] - bbox[..., :2] - 1
        elif bbox_format == 'xyxy':
            bbox_flipped[...] = [w, h, w, h] - bbox - 1
            bbox_flipped = np.concatenate(
                (bbox_flipped[..., 2:], bbox_flipped[..., :2]), axis=-1)

    return bbox_flipped


def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Calculate the affine transformation matrix under the unbiased
    constraint. See `UDP (CVPR 2020)`_ for details.

    Note:

        - The bbox number: N

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image

    Returns:
        np.ndarray: A 2x3 transformation matrix

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (-0.5 * input_size[0] * math.cos(rot_rad) +
                                0.5 * input_size[1] * math.sin(rot_rad) +
                                0.5 * scale[0])
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (-0.5 * input_size[0] * math.sin(rot_rad) -
                                0.5 * input_size[1] * math.cos(rot_rad) +
                                0.5 * scale[1])
    return warp_mat


def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat


def get_pers_warp_matrix(center: np.ndarray, translate: np.ndarray,
                         scale: float, rot: float,
                         shear: np.ndarray) -> np.ndarray:
    """Compute a perspective warp matrix based on specified transformations.

    Args:
        center (np.ndarray): Center of the transformation.
        translate (np.ndarray): Translation vector.
        scale (float): Scaling factor.
        rot (float): Rotation angle in degrees.
        shear (np.ndarray): Shearing angles in degrees along x and y axes.

    Returns:
        np.ndarray: Perspective warp matrix.

    Example:
        >>> center = np.array([0, 0])
        >>> translate = np.array([10, 20])
        >>> scale = 1.2
        >>> rot = 30.0
        >>> shear = np.array([15.0, 0.0])
        >>> warp_matrix = get_pers_warp_matrix(center, translate,
                                               scale, rot, shear)
    """
    translate_mat = np.array([[1, 0, translate[0] + center[0]],
                              [0, 1, translate[1] + center[1]], [0, 0, 1]],
                             dtype=np.float32)

    shear_x = math.radians(shear[0])
    shear_y = math.radians(shear[1])
    shear_mat = np.array([[1, np.tan(shear_x), 0], [np.tan(shear_y), 1, 0],
                          [0, 0, 1]],
                         dtype=np.float32)

    rotate_angle = math.radians(rot)
    rotate_mat = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle), 0],
                           [np.sin(rotate_angle),
                            np.cos(rotate_angle), 0], [0, 0, 1]],
                          dtype=np.float32)

    scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
                         dtype=np.float32)

    recover_center_mat = np.array([[1, 0, -center[0]], [0, 1, -center[1]],
                                   [0, 0, 1]],
                                  dtype=np.float32)

    warp_matrix = np.dot(
        np.dot(
            np.dot(np.dot(translate_mat, shear_mat), rotate_mat), scale_mat),
        recover_center_mat)

    return warp_matrix


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c
