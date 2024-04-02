# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class Instances:
    keypoints: List[List[Tuple[int, int]]]
    keypoint_scores: List[List[float]]


class FastVisualizer:
    """MMPose Fast Visualizer.

    A simple yet fast visualizer for video/webcam inference.

    Args:
        metainfo (dict): pose meta information
        radius (int, optional): Keypoint radius for visualization.
            Defaults to 6.
        line_width (int, optional): Link width for visualization.
            Defaults to 3.
        kpt_thr (float, optional): Threshold for keypoints' confidence score,
            keypoints with score below this value will not be drawn.
            Defaults to 0.3.
    """

    def __init__(self,
                 metainfo: Dict,
                 radius: Optional[int] = 6,
                 line_width: Optional[int] = 3,
                 kpt_thr: Optional[float] = 0.3):
        self.radius = radius
        self.line_width = line_width
        self.kpt_thr = kpt_thr

        self.keypoint_id2name = metainfo.get('keypoint_id2name', None)
        self.keypoint_name2id = metainfo.get('keypoint_name2id', None)
        self.keypoint_colors = metainfo.get('keypoint_colors',
                                            [(255, 255, 255)])
        self.skeleton_links = metainfo.get('skeleton_links', None)
        self.skeleton_link_colors = metainfo.get('skeleton_link_colors', None)

    def draw_pose(self, img: np.ndarray, instances: Instances):
        """Draw pose estimations on the given image.

        This method draws keypoints and skeleton links on the input image
        using the provided instances.

        Args:
            img (numpy.ndarray): The input image on which to
                draw the pose estimations.
            instances (object): An object containing detected instances'
                information, including keypoints and keypoint_scores.

        Returns:
            None: The input image will be modified in place.
        """

        if instances is None:
            print('no instance detected')
            return

        keypoints = instances.keypoints
        scores = instances.keypoint_scores

        for kpts, score in zip(keypoints, scores):
            for sk_id, sk in enumerate(self.skeleton_links):
                if score[sk[0]] < self.kpt_thr or score[sk[1]] < self.kpt_thr:
                    # skip the link that should not be drawn
                    continue

                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                color = self.skeleton_link_colors[sk_id].tolist()
                cv2.line(img, pos1, pos2, color, thickness=self.line_width)

            for kid, kpt in enumerate(kpts):
                if score[kid] < self.kpt_thr:
                    # skip the point that should not be drawn
                    continue

                x_coord, y_coord = int(kpt[0]), int(kpt[1])

                color = self.keypoint_colors[kid].tolist()
                cv2.circle(img, (int(x_coord), int(y_coord)), self.radius,
                           color, -1)
                cv2.circle(img, (int(x_coord), int(y_coord)), self.radius,
                           (255, 255, 255))

    def draw_points(self, img: np.ndarray, instances: Union[Instances, Dict,
                                                            np.ndarray]):
        """Draw points on the given image.

        This method draws keypoints on the input image
        using the provided instances.

        Args:
            img (numpy.ndarray): The input image on which to
                draw the keypoints.
            instances (object|dict|np.ndarray):
                An object containing keypoints,
                or a dict containing 'keypoints',
                or a np.ndarray in shape of
                (Instance_num, Point_num, Point_dim)

        Returns:
            None: The input image will be modified in place.
        """

        if instances is None:
            print('no instance detected')
            return

        # support different types of keypoints inputs
        if hasattr(instances, 'keypoints'):
            keypoints = instances.keypoints
        elif isinstance(instances, dict) and 'keypoints' in instances:
            keypoints = instances['keypoints']
        elif isinstance(instances, np.ndarray):
            shape = instances.shape
            assert shape[-1] == 2, 'only support 2-dim point!'
            if len(shape) == 2:
                keypoints = instances[None]
            elif len(shape) == 3:
                pass
            else:
                raise ValueError('input keypoints should be in shape of'
                                 '(Instance_num, Point_num, Point_dim)')
        else:
            raise ValueError('The keypoints should be:'
                             'object containing keypoints,'
                             "or a dict containing 'keypoints',"
                             'or a np.ndarray in shape of'
                             '(Instance_num, Point_num, Point_dim)')

        if len(self.keypoint_colors) < len(keypoints[0]):
            repeat_num = len(keypoints[0]) - len(self.keypoint_colors)
            self.keypoint_colors += [(255, 255, 255)] * repeat_num
        self.keypoint_colors = np.array(self.keypoint_colors)

        for kpts in keypoints:
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord = int(kpt[0]), int(kpt[1])

                color = self.keypoint_colors[kid].tolist()
                cv2.circle(img, (int(x_coord), int(y_coord)), self.radius,
                           color, -1)
                cv2.circle(img, (int(x_coord), int(y_coord)), self.radius,
                           (255, 255, 255))
