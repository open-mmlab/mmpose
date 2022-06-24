# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmengine import Visualizer
from mmengine.data import InstanceData, PixelData
from mmengine.dist import master_only

from mmpose.core import PoseDataSample
from mmpose.registry import VISUALIZERS


@VISUALIZERS.register_module()
class PoseLocalVisualizer(Visualizer):
    """MMPose Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.22222
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        kpt_color (str, tuple(int), optional): Color of keypoints.
            The tuple of color should be in BGR order.
            Defaults to None.
        link_color (str, tuple(int), optional): Color of skeleton.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The width of lines.
            Defaults to 1.
        radius (int, float): The radius of keypoints.
            Defaults to 4.
        show_keypoint_weight (bool): Whether to adjust the transparency
            of keypoints according to their score. Defaults to False.
        alpha (int, float): The transparency of bboxes.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> from mmengine.data import InstanceData
        >>> from mmpose.core import PoseDataSample
        >>> from mmpose.core import PoseLocalVisualizer

        >>> pose_local_visualizer = PoseLocalVisualizer(radius=1)
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                          [8, 8]]])
        >>> gt_pose_data_sample = PoseDataSample()
        >>> gt_pose_data_sample.gt_instances = gt_instances
        >>> pose_local_visualizer.dataset_meta = {'SKELETON': [[0, 1], [1, 2],
        ...                                                    [2, 3]]}
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample)
        >>> pose_local_visualizer.add_datasample(
        ...                       'image', image, gt_pose_data_sample,
        ...                        out_file='out_file.jpg')
        >>> pose_local_visualizer.add_datasample(
        ...                        'image', image, gt_pose_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                       [8, 8]]])
        >>> pred_instances.score = np.array([0.8, 1, 0.9, 1])
        >>> pred_pose_data_sample = PoseDataSample()
        >>> pred_pose_data_sample.pred_instances = pred_instances
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample,
        ...                         pred_pose_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = 'green',
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 kpt_color: Optional[Union[str, Tuple[int]]] = None,
                 link_color: Optional[Union[str, Tuple[int]]] = None,
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 1,
                 radius: Union[int, float] = 4,
                 show_keypoint_weight: bool = False,
                 alpha: float = 0.8):
        super().__init__(name, image, vis_backends, save_dir)
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight
        # Set default value. When calling
        # `PoseLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances_bbox(self, image: np.ndarray,
                             instances: ['InstanceData']) -> np.ndarray:
        """Draw bounding boxes of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)
        if 'bboxes' in instances:
            bboxes = instances.bboxes
            self.draw_bboxes(
                bboxes,
                edge_colors=self.bbox_color,
                alpha=self.alpha,
                line_widths=self.line_width)

        return self.get_image()

    def _draw_instances_kpts(self,
                             image: np.ndarray,
                             instances: ['InstanceData'],
                             skeleton: Optional[Union[List, Tuple]] = None,
                             kpt_score_thr: float = 0.3):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            skeleton (tuple or list, optional): Composed of keypoint index
                pairs that need to be plotted.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        self.set_image(image)
        img_h, img_w, _ = image.shape

        if 'keypoints' in instances:
            keypoints = instances.keypoints
            if 'scores' in instances and self.show_keypoint_weight:
                scores = instances.scores
            else:
                scores = [np.ones(len(kpts)) for kpts in keypoints]

            for kpts, score in zip(keypoints, scores):
                kpts = np.array(kpts, copy=False)

                # draw each point on image
                if self.kpt_color is not None:
                    assert len(self.kpt_color) == len(kpts)

                    for kid, kpt in enumerate(kpts):

                        if score[kid] < kpt_score_thr or self.kpt_color[
                                kid] is None:
                            # skip the point that should not be drawn
                            continue

                        color = tuple(int(c) for c in self.kpt_color[kid])
                        transparency = max(0, min(1, score[kid]))
                        self.draw_circles(
                            kpt,
                            radius=np.array([self.radius]),
                            face_colors=color,
                            edge_colors=color,
                            alpha=transparency,
                            line_widths=self.radius)

                # draw links
                if skeleton is not None and self.link_color is not None:
                    assert len(self.link_color) == len(skeleton)

                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h
                                or score[sk[0]] < kpt_score_thr
                                or score[sk[1]] < kpt_score_thr
                                or self.link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue
                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = tuple(int(c) for c in self.link_color[sk_id])
                        if self.show_keypoint_weight:

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
                            transparency = max(
                                0, min(1, 0.5 * (score[sk[0]] + score[sk[1]])))
                            self.draw_polygons(
                                polygons,
                                edge_colors=color,
                                face_colors=color,
                                alpha=transparency)

                        else:
                            self.draw_lines(
                                X, Y, color, line_widths=self.line_width)

        return self.get_image()

    def _draw_instance_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[Union[np.ndarray]] = None,
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
                pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        out_image = self.draw_featmap(fields.heatmaps, overlaid_image)
        return out_image

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            gt_sample: Optional['PoseDataSample'] = None,
            pred_sample: Optional['PoseDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            draw_heatmap: bool = False,
            draw_bbox: bool = False,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            kpt_score_thr: float = 0.3,
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
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`PoseDataSample`, optional): GT PoseDataSample.
                Defaults to None.
            pred_sample (:obj:`PoseDataSample`, optional): Prediction
                PoseDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction PoseDataSample.
                Defaults to True.
            draw_bbox (bool): Whether to draw bounding boxes. Default to False.
            draw_heatmap (bool): Whether to draw heatmaps. Defaults to False.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        skeleton = self.dataset_meta.get('SKELETON', None)

        gt_img_data = None
        pred_img_data = None

        if draw_gt and gt_sample is not None:
            gt_img_data = image.copy()
            gt_img_heatmap = None
            if 'gt_instances' in gt_sample:
                gt_img_data = self._draw_instances_kpts(
                    gt_img_data, gt_sample.gt_instances, skeleton,
                    kpt_score_thr)
            if 'gt_instances' in gt_sample and draw_bbox:
                gt_img_data = self._draw_instances_bbox(
                    gt_img_data, gt_sample.gt_instances)

            if 'gt_fields' in gt_sample and draw_heatmap:
                gt_img_heatmap = self._draw_instance_heatmap(
                    gt_sample.gt_fields, image)
                gt_img_data = np.concatenate((gt_img_data, gt_img_heatmap),
                                             axis=0)

        if draw_pred and pred_sample is not None:
            pred_img_data = image.copy()
            pred_img_heatmap = None
            if 'pred_instances' in pred_sample:
                pred_img_data = self._draw_instances_kpts(
                    pred_img_data, pred_sample.pred_instances, skeleton,
                    kpt_score_thr)
            if 'pred_instances' in pred_sample and draw_bbox:
                pred_img_data = self._draw_instances_bbox(
                    pred_img_data, pred_sample.pred_instances)
            if 'pred_fields' in pred_sample and draw_heatmap:
                pred_img_heatmap = self._draw_instance_heatmap(
                    pred_sample.pred_fields, image)
                pred_img_data = np.concatenate(
                    (pred_img_data, pred_img_heatmap), axis=0)

        if gt_img_data is not None and pred_img_data is not None:
            if gt_img_heatmap is None and pred_img_heatmap is not None:
                gt_img_data = np.concatenate((gt_img_data, image), axis=0)
            elif gt_img_heatmap is not None and pred_img_heatmap is None:
                pred_img_data = np.concatenate((pred_img_data, image), axis=0)

            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)

        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, drawn_img, step)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
