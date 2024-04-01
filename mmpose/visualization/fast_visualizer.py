# Copyright (c) OpenMMLab. All rights reserved.
import cv2


class FastVisualizer:
    """MMPose Fast Visualizer.

    A simple yet fast visualizer for video/webcam inference.

    Args:
        metainfo (dict): pose meta information
        radius (int, optional)): Keypoint radius for visualization.
            Defaults to 6.
        line_width (int, optional): Link width for visualization.
            Defaults to 3.
        kpt_thr (float, optional): Threshold for keypoints' confidence score,
            keypoints with score below this value will not be drawn.
            Defaults to 0.3.
    """

    def __init__(self, metainfo, radius=6, line_width=3, kpt_thr=0.3):
        self.radius = radius
        self.line_width = line_width
        self.kpt_thr = kpt_thr

        self.keypoint_id2name = metainfo['keypoint_id2name']
        self.keypoint_name2id = metainfo['keypoint_name2id']
        self.keypoint_colors = metainfo['keypoint_colors']
        self.skeleton_links = metainfo['skeleton_links']
        self.skeleton_link_colors = metainfo['skeleton_link_colors']

    def draw_pose(self, img, instances):
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
