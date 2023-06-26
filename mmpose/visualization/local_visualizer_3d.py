# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from mmengine.dist import master_only
from mmengine.structures import InstanceData

from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample
from . import PoseLocalVisualizer


@VISUALIZERS.register_module()
class Pose3dLocalVisualizer(PoseLocalVisualizer):
    """MMPose 3d Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to ``None``
        vis_backends (list, optional): Visual backend config list. Defaults to
            ``None``
        save_dir (str, optional): Save file dir for all storage backends.
            If it is ``None``, the backend storage will not save any data.
            Defaults to ``None``
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to ``'green'``
        kpt_color (str, tuple(tuple(int)), optional): Color of keypoints.
            The tuple of color should be in BGR order. Defaults to ``'red'``
        link_color (str, tuple(tuple(int)), optional): Color of skeleton.
            The tuple of color should be in BGR order. Defaults to ``None``
        line_width (int, float): The width of lines. Defaults to 1
        radius (int, float): The radius of keypoints. Defaults to 4
        show_keypoint_weight (bool): Whether to adjust the transparency
            of keypoints according to their score. Defaults to ``False``
        alpha (int, float): The transparency of bboxes. Defaults to ``0.8``
        det_kpt_color (str, tuple(tuple(int)), optional): Keypoints color
             info for detection. Defaults to ``None``
        det_dataset_skeleton (list): Skeleton info for detection. Defaults to
            ``None``
        det_dataset_link_color (list): Link color for detection. Defaults to
            ``None``
    """

    def __init__(
            self,
            name: str = 'visualizer',
            image: Optional[np.ndarray] = None,
            vis_backends: Optional[Dict] = None,
            save_dir: Optional[str] = None,
            bbox_color: Optional[Union[str, Tuple[int]]] = 'green',
            kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = 'red',
            link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
            text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
            skeleton: Optional[Union[List, Tuple]] = None,
            line_width: Union[int, float] = 1,
            radius: Union[int, float] = 3,
            show_keypoint_weight: bool = False,
            backend: str = 'opencv',
            alpha: float = 0.8,
            det_kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
            det_dataset_skeleton: Optional[Union[str,
                                                 Tuple[Tuple[int]]]] = None,
            det_dataset_link_color: Optional[np.ndarray] = None):
        super().__init__(name, image, vis_backends, save_dir, bbox_color,
                         kpt_color, link_color, text_color, skeleton,
                         line_width, radius, show_keypoint_weight, backend,
                         alpha)
        self.det_kpt_color = det_kpt_color
        self.det_dataset_skeleton = det_dataset_skeleton
        self.det_dataset_link_color = det_dataset_link_color

    def _draw_3d_data_samples(
        self,
        image: np.ndarray,
        pose_samples: PoseDataSample,
        draw_gt: bool = True,
        kpt_thr: float = 0.3,
        num_instances=-1,
        axis_azimuth: float = 70,
        axis_limit: float = 1.7,
        axis_dist: float = 10.0,
        axis_elev: float = 15.0,
    ):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            num_instances (int): Number of instances to be shown in 3D. If
                smaller than 0, all the instances in the pose_result will be
                shown. Otherwise, pad or truncate the pose_result to a length
                of num_instances.
            axis_azimuth (float): axis azimuth angle for 3D visualizations.
            axis_dist (float): axis distance for 3D visualizations.
            axis_elev (float): axis elevation view angle for 3D visualizations.
            axis_limit (float): The axis limit to visualize 3d pose. The xyz
                range will be set as:
                - x: [x_c - axis_limit/2, x_c + axis_limit/2]
                - y: [y_c - axis_limit/2, y_c + axis_limit/2]
                - z: [0, axis_limit]
                Where x_c, y_c is the mean value of x and y coordinates

        Returns:
            Tuple(np.ndarray): the drawn image which channel is RGB.
        """
        vis_height, vis_width, _ = image.shape

        if 'pred_instances' in pose_samples:
            pred_instances = pose_samples.pred_instances
        else:
            pred_instances = InstanceData()
        if num_instances < 0:
            if 'keypoints' in pred_instances:
                num_instances = len(pred_instances)
            else:
                num_instances = 0
        else:
            if len(pred_instances) > num_instances:
                pred_instances_ = InstanceData()
                for k in pred_instances.keys():
                    new_val = pred_instances[k][:num_instances]
                    pred_instances_.set_field(new_val, k)
                pred_instances = pred_instances_
            elif num_instances < len(pred_instances):
                num_instances = len(pred_instances)

        num_fig = num_instances
        if draw_gt:
            vis_width *= 2
            num_fig *= 2

        plt.ioff()
        fig = plt.figure(
            figsize=(vis_width * num_instances * 0.01, vis_height * 0.01))

        def _draw_3d_instances_kpts(keypoints,
                                    scores,
                                    keypoints_visible,
                                    fig_idx,
                                    title=None):

            for idx, (kpts, score, visible) in enumerate(
                    zip(keypoints, scores, keypoints_visible)):

                valid = np.logical_and(score >= kpt_thr,
                                       np.any(~np.isnan(kpts), axis=-1))

                ax = fig.add_subplot(
                    1, num_fig, fig_idx * (idx + 1), projection='3d')
                ax.view_init(elev=axis_elev, azim=axis_azimuth)
                ax.set_zlim3d([0, axis_limit])
                ax.set_aspect('auto')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.scatter([0], [0], [0], marker='o', color='red')
                if title:
                    ax.set_title(f'{title} ({idx})')
                ax.dist = axis_dist

                x_c = np.mean(kpts[valid, 0]) if valid.any() else 0
                y_c = np.mean(kpts[valid, 1]) if valid.any() else 0

                ax.set_xlim3d([x_c - axis_limit / 2, x_c + axis_limit / 2])
                ax.set_ylim3d([y_c - axis_limit / 2, y_c + axis_limit / 2])

                kpts = np.array(kpts, copy=False)

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                kpts = kpts[valid]
                x_3d, y_3d, z_3d = np.split(kpts[:, :3], [1, 2], axis=1)

                kpt_color = kpt_color[valid][..., ::-1] / 255.

                ax.scatter(x_3d, y_3d, z_3d, marker='o', color=kpt_color)

                for kpt_idx in range(len(x_3d)):
                    ax.text(x_3d[kpt_idx][0], y_3d[kpt_idx][0],
                            z_3d[kpt_idx][0], str(kpt_idx))

                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        sk_indices = [_i for _i in sk]
                        xs_3d = kpts[sk_indices, 0]
                        ys_3d = kpts[sk_indices, 1]
                        zs_3d = kpts[sk_indices, 2]
                        kpt_score = score[sk_indices]
                        if kpt_score.min() > kpt_thr:
                            # matplotlib uses RGB color in [0, 1] value range
                            _color = link_color[sk_id][::-1] / 255.
                            ax.plot(
                                xs_3d, ys_3d, zs_3d, color=_color, zdir='z')

        if 'keypoints' in pred_instances:
            keypoints = pred_instances.get('keypoints',
                                           pred_instances.keypoints)

            if 'keypoint_scores' in pred_instances:
                scores = pred_instances.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if 'keypoints_visible' in pred_instances:
                keypoints_visible = pred_instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            _draw_3d_instances_kpts(keypoints, scores, keypoints_visible, 1,
                                    'Prediction')

        if draw_gt and 'gt_instances' in pose_samples:
            gt_instances = pose_samples.gt_instances
            if 'lifting_target' in gt_instances:
                keypoints = gt_instances.get('lifting_target',
                                             gt_instances.lifting_target)
                scores = np.ones(keypoints.shape[:-1])

                if 'lifting_target_visible' in gt_instances:
                    keypoints_visible = gt_instances.lifting_target_visible
                else:
                    keypoints_visible = np.ones(keypoints.shape[:-1])

                _draw_3d_instances_kpts(keypoints, scores, keypoints_visible,
                                        2, 'Ground Truth')

        # convert figure to numpy array
        fig.tight_layout()
        fig.canvas.draw()

        pred_img_data = fig.canvas.tostring_rgb()
        pred_img_data = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8)

        if not pred_img_data.any():
            pred_img_data = np.full((vis_height, vis_width, 3), 255)
        else:
            pred_img_data = pred_img_data.reshape(vis_height,
                                                  vis_width * num_instances,
                                                  -1)

        plt.close(fig)

        return pred_img_data

    def _draw_instances_kpts(self,
                             image: np.ndarray,
                             instances: InstanceData,
                             kpt_thr: float = 0.3,
                             show_kpt_idx: bool = False,
                             skeleton_style: str = 'mmpose'):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        self.set_image(image)
        img_h, img_w, _ = image.shape

        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)

            if 'keypoint_scores' in instances:
                scores = instances.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            if skeleton_style == 'openpose':
                keypoints_info = np.concatenate(
                    (keypoints, scores[..., None], keypoints_visible[...,
                                                                     None]),
                    axis=-1)
                # compute neck joint
                neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
                # neck score when visualizing pred
                neck[:, 2:4] = np.logical_and(
                    keypoints_info[:, 5, 2:4] > kpt_thr,
                    keypoints_info[:, 6, 2:4] > kpt_thr).astype(int)
                new_keypoints_info = np.insert(
                    keypoints_info, 17, neck, axis=1)

                mmpose_idx = [
                    17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
                ]
                openpose_idx = [
                    1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
                ]
                new_keypoints_info[:, openpose_idx] = \
                    new_keypoints_info[:, mmpose_idx]
                keypoints_info = new_keypoints_info

                keypoints, scores, keypoints_visible = keypoints_info[
                    ..., :2], keypoints_info[..., 2], keypoints_info[..., 3]

            kpt_color = self.kpt_color
            if self.det_kpt_color is not None:
                kpt_color = self.det_kpt_color

            for kpts, score, visible in zip(keypoints, scores,
                                            keypoints_visible):
                kpts = np.array(kpts, copy=False)

                if kpt_color is None or isinstance(kpt_color, str):
                    kpt_color = [kpt_color] * len(kpts)
                elif len(kpt_color) == len(kpts):
                    kpt_color = kpt_color
                else:
                    raise ValueError(f'the length of kpt_color '
                                     f'({len(kpt_color)}) does not matches '
                                     f'that of keypoints ({len(kpts)})')

                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if score[kid] < kpt_thr or not visible[
                            kid] or kpt_color[kid] is None:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, score[kid]))
                    self.draw_circles(
                        kpt,
                        radius=np.array([self.radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=self.radius)
                    if show_kpt_idx:
                        self.draw_texts(
                            str(kid),
                            kpt,
                            colors=color,
                            font_sizes=self.radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')

                # draw links
                skeleton = self.skeleton
                if self.det_dataset_skeleton is not None:
                    skeleton = self.det_dataset_skeleton
                link_color = self.link_color
                if self.det_dataset_link_color is not None:
                    link_color = self.det_dataset_link_color
                if skeleton is not None and link_color is not None:
                    if link_color is None or isinstance(link_color, str):
                        link_color = [link_color] * len(skeleton)
                    elif len(link_color) == len(skeleton):
                        link_color = link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(link_color)}) does not matches '
                            f'that of skeleton ({len(skeleton)})')

                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                        if not (visible[sk[0]] and visible[sk[1]]):
                            continue

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h or score[sk[0]] < kpt_thr
                                or score[sk[1]] < kpt_thr
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue
                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0, min(1, 0.5 * (score[sk[0]] + score[sk[1]])))

                        if skeleton_style == 'openpose':
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygons = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            self.draw_polygons(
                                polygons,
                                edge_colors=color,
                                face_colors=color,
                                alpha=transparency)

                        else:
                            self.draw_lines(
                                X, Y, color, line_widths=self.line_width)

        return self.get_image()

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: PoseDataSample,
                       det_data_sample: Optional[PoseDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_2d: bool = True,
                       draw_bbox: bool = False,
                       show_kpt_idx: bool = False,
                       skeleton_style: str = 'mmpose',
                       num_instances: int = -1,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       kpt_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier
            image (np.ndarray): The image to draw
            data_sample (:obj:`PoseDataSample`): The 3d data sample
                to visualize
            det_data_sample (:obj:`PoseDataSample`, optional): The 2d detection
                data sample to visualize
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            draw_pred (bool): Whether to draw Prediction PoseDataSample.
                Defaults to ``True``
            draw_2d (bool): Whether to draw 2d detection results. Defaults to
                ``True``
            draw_bbox (bool): Whether to draw bounding boxes. Default to
                ``False``
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``
            num_instances (int): Number of instances to be shown in 3D. If
                smaller than 0, all the instances in the pose_result will be
                shown. Otherwise, pad or truncate the pose_result to a length
                of num_instances. Defaults to -1
            show (bool): Whether to display the drawn image. Default to
                ``False``
            wait_time (float): The interval of show (s). Defaults to 0
            out_file (str): Path to output file. Defaults to ``None``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            step (int): Global step value to record. Defaults to 0
        """

        det_img_data = None
        gt_img_data = None

        if draw_2d:
            det_img_data = image.copy()

            # draw bboxes & keypoints
            if 'pred_instances' in det_data_sample:
                det_img_data = self._draw_instances_kpts(
                    det_img_data, det_data_sample.pred_instances, kpt_thr,
                    show_kpt_idx, skeleton_style)
                if draw_bbox:
                    det_img_data = self._draw_instances_bbox(
                        det_img_data, det_data_sample.pred_instances)

        pred_img_data = self._draw_3d_data_samples(
            image.copy(),
            data_sample,
            draw_gt=draw_gt,
            num_instances=num_instances)

        # merge visualization results
        if det_img_data is not None and gt_img_data is not None:
            drawn_img = np.concatenate(
                (det_img_data, pred_img_data, gt_img_data), axis=1)
        elif det_img_data is not None:
            drawn_img = np.concatenate((det_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = np.concatenate((det_img_data, gt_img_data), axis=1)
        else:
            drawn_img = pred_img_data

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            # save drawn_img to backends
            self.add_image(name, drawn_img, step)

        return self.get_image()
