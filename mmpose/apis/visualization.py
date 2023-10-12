# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Union

import mmcv
import numpy as np
from mmengine.structures import InstanceData

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


def visualize(
    img: Union[np.ndarray, str],
    keypoints: np.ndarray,
    keypoint_score: np.ndarray = None,
    metainfo: Union[str, dict] = None,
    visualizer: PoseLocalVisualizer = None,
    show_kpt_idx: bool = False,
    skeleton_style: str = 'mmpose',
    show: bool = False,
    kpt_thr: float = 0.3,
):
    """Visualize 2d keypoints on an image.

    Args:
        img (str | np.ndarray): The image to be displayed.
        keypoints (np.ndarray): The keypoint to be displayed.
        keypoint_score (np.ndarray): The score of each keypoint.
        metainfo (str | dict): The metainfo of dataset.
        visualizer (PoseLocalVisualizer): The visualizer.
        show_kpt_idx (bool): Whether to show the index of keypoints.
        skeleton_style (str): Skeleton style. Options are 'mmpose' and
            'openpose'.
        show (bool): Whether to show the image.
        wait_time (int): Value of waitKey param.
        kpt_thr (float): Keypoint threshold.
    """
    assert skeleton_style in [
        'mmpose', 'openpose'
    ], (f'Only support skeleton style in {["mmpose", "openpose"]}, ')

    if visualizer is None:
        visualizer = PoseLocalVisualizer()
    else:
        visualizer = deepcopy(visualizer)

    if isinstance(metainfo, str):
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
    elif isinstance(metainfo, dict):
        metainfo = parse_pose_metainfo(metainfo)

    if metainfo is not None:
        visualizer.set_dataset_meta(metainfo, skeleton_style=skeleton_style)

    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if keypoint_score is None:
        keypoint_score = np.ones(keypoints.shape[0])

    tmp_instances = InstanceData()
    tmp_instances.keypoints = keypoints
    tmp_instances.keypoint_score = keypoint_score

    tmp_datasample = PoseDataSample()
    tmp_datasample.pred_instances = tmp_instances

    visualizer.add_datasample(
        'visualization',
        img,
        tmp_datasample,
        show_kpt_idx=show_kpt_idx,
        skeleton_style=skeleton_style,
        show=show,
        wait_time=0,
        kpt_thr=kpt_thr)

    return visualizer.get_image()
