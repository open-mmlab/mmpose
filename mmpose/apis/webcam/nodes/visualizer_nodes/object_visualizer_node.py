# Copyright (c) OpenMMLab. All rights reserved.
import math
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np

from ...utils import FrameMessage
from ..base_visualizer_node import BaseVisualizerNode
from ..registry import NODES


def imshow_bboxes(img,
                  bboxes,
                  labels=None,
                  colors='green',
                  text_color='white',
                  thickness=1,
                  font_scale=0.5):
    """Draw bboxes with labels (optional) on an image. This is a wrapper of
    mmcv.imshow_bboxes.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): ndarray of shape (k, 4), each row is a bbox in
            format [x1, y1, x2, y2].
        labels (str or list[str], optional): labels of each bbox.
        colors (list[str or tuple or :obj:`Color`]): A list of colors.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """

    # adapt to mmcv.imshow_bboxes input format
    bboxes = np.split(
        bboxes, bboxes.shape[0], axis=0) if bboxes.shape[0] > 0 else []
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [mmcv.color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    img = mmcv.imshow_bboxes(
        img,
        bboxes,
        colors,
        top_k=-1,
        thickness=thickness,
        show=False,
        out_file=None)

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels for _ in range(len(bboxes))]
        assert len(labels) == len(bboxes)

        for bbox, label, color in zip(bboxes, labels, colors):
            if label is None:
                continue
            bbox_int = bbox[0, :4].astype(np.int32)
            # roughly estimate the proper font size
            text_size, text_baseline = cv2.getTextSize(label,
                                                       cv2.FONT_HERSHEY_DUPLEX,
                                                       font_scale, thickness)
            text_x1 = bbox_int[0]
            text_y1 = max(0, bbox_int[1] - text_size[1] - text_baseline)
            text_x2 = bbox_int[0] + text_size[0]
            text_y2 = text_y1 + text_size[1] + text_baseline
            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color,
                          cv2.FILLED)
            cv2.putText(img, label, (text_x1, text_y2 - text_baseline),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale,
                        mmcv.color_val(text_color), thickness)

    return img


def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    img = mmcv.imread(img)
    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                        int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(
                        0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img


@NODES.register_module()
class ObjectVisualizerNode(BaseVisualizerNode):
    """Visualize the bounding box and keypoints of objects.

    Args:
        name (str): The node name (also thread name)
        input_buffer (str): The name of the input buffer
        output_buffer (str|list): The name(s) of the output buffer(s)
        enable_key (str|int, optional): Set a hot-key to toggle enable/disable
            of the node. If an int value is given, it will be treated as an
            ascii code of a key. Please note: (1) If ``enable_key`` is set,
            the ``bypass()`` method need to be overridden to define the node
            behavior when disabled; (2) Some hot-keys are reserved for
            particular use. For example: 'q', 'Q' and 27 are used for exiting.
            Default: ``None``
        enable (bool): Default enable/disable status. Default: ``True``
        show_bbox (bool): Set ``True`` to show the bboxes of detection
            objects. Default: ``True``
        show_keypoint (bool): Set ``True`` to show the pose estimation
            results. Default: ``True``
        must_have_bbox (bool): Only show objects with keypoints.
            Default: ``False``
        kpt_thr (float): The threshold of keypoint score. Default: 0.3
        radius (int): The radius of keypoint. Default: 4
        thickness (int): The thickness of skeleton. Default: 2
        bbox_color (str|tuple|dict): The color of bboxes. If a single color is
            given (a str like 'green' or a BGR tuple like (0, 255, 0)), it
            will be used for all bboxes. If a dict is given, it will be used
            as a map from class labels to bbox colors. If not given, a default
            color map will be used. Default: ``None``

    Example::
        >>> cfg = dict(
        ...    type='ObjectVisualizerNode',
        ...    name='object visualizer',
        ...    enable_key='v',
        ...    enable=True,
        ...    show_bbox=True,
        ...    must_have_keypoint=False,
        ...    show_keypoint=True,
        ...    input_buffer='frame',
        ...    output_buffer='vis')

        >>> from mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    default_bbox_color = {
        'person': (148, 139, 255),
        'cat': (255, 255, 0),
        'dog': (255, 255, 0),
    }

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 show_bbox: bool = True,
                 show_keypoint: bool = True,
                 must_have_keypoint: bool = False,
                 kpt_thr: float = 0.3,
                 radius: int = 4,
                 thickness: int = 2,
                 bbox_color: Optional[Union[str, Tuple, Dict]] = 'green'):

        super().__init__(
            name=name,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            enable_key=enable_key,
            enable=enable)

        self.kpt_thr = kpt_thr
        self.bbox_color = bbox_color
        self.show_bbox = show_bbox
        self.show_keypoint = show_keypoint
        self.must_have_keypoint = must_have_keypoint
        self.radius = radius
        self.thickness = thickness

    def _draw_bbox(self, canvas: np.ndarray, input_msg: FrameMessage):
        """Draw object bboxes."""

        if self.must_have_keypoint:
            objects = input_msg.get_objects(
                lambda x: 'bbox' in x and 'keypoints' in x)
        else:
            objects = input_msg.get_objects(lambda x: 'bbox' in x)
        # return if there is no detected objects
        if not objects:
            return canvas

        bboxes = [obj['bbox'] for obj in objects]
        labels = [obj.get('label', None) for obj in objects]
        default_color = (0, 255, 0)

        # Get bbox colors
        if isinstance(self.bbox_color, dict):
            colors = [
                self.bbox_color.get(label, default_color) for label in labels
            ]
        else:
            colors = self.bbox_color

        imshow_bboxes(
            canvas,
            np.vstack(bboxes),
            labels=labels,
            colors=colors,
            text_color='white',
            font_scale=0.5)

        return canvas

    def _draw_keypoint(self, canvas: np.ndarray, input_msg: FrameMessage):
        """Draw object keypoints."""
        objects = input_msg.get_objects(lambda x: 'pose_model_cfg' in x)

        # return if there is no object with keypoints
        if not objects:
            return canvas

        for model_cfg, group in groupby(objects,
                                        lambda x: x['pose_model_cfg']):
            dataset_info = objects[0]['dataset_meta']
            keypoints = [
                np.concatenate(
                    (obj['keypoints'], obj['keypoint_scores'][:, None]),
                    axis=1) for obj in group
            ]
            imshow_keypoints(
                canvas,
                keypoints,
                skeleton=dataset_info['skeleton_links'],
                kpt_score_thr=self.kpt_thr,
                pose_kpt_color=dataset_info['keypoint_colors'],
                pose_link_color=dataset_info['skeleton_link_colors'],
                radius=self.radius,
                thickness=self.thickness)

        return canvas

    def draw(self, input_msg: FrameMessage) -> np.ndarray:
        canvas = input_msg.get_image()

        if self.show_bbox:
            canvas = self._draw_bbox(canvas, input_msg)

        if self.show_keypoint:
            canvas = self._draw_keypoint(canvas, input_msg)

        return canvas
