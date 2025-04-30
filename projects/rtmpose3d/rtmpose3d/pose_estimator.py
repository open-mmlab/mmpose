from itertools import zip_longest
from typing import Optional

import numpy as np

from mmpose.models.pose_estimators import TopdownPoseEstimator
from mmpose.registry import MODELS
from mmpose.utils.typing import InstanceList, PixelDataList, SampleList


@MODELS.register_module()
class TopdownPoseEstimator3D(TopdownPoseEstimator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # a default camera parameter for 3D pose estimation
        self.camera_param = {
            'c': [512.54150496, 515.45148698],
            'f': [1145.04940459, 1143.78109572],
        }

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)
        mode = self.test_cfg.get('mode', '3d')
        assert mode in ['2d', '3d', 'vis']
        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']
            keypoints_3d = pred_instances.keypoints
            keypoints_simcc = pred_instances.keypoints_simcc

            # convert keypoints from input space to image space
            keypoints_2d = keypoints_3d[..., :2].copy()
            keypoints_2d = keypoints_2d / input_size * input_scale \
                + input_center - 0.5 * input_scale

            # convert keypoints from image space to camera space
            if gt_instances.get('camera_params', None) is not None:
                camera_params = gt_instances.camera_params[0]
                f = np.array(camera_params['f'])
                c = np.array(camera_params['c'])
            else:
                f = np.array([1145.04940459, 1143.78109572])
                c = np.array(data_sample.ori_shape) / 2
            kpts_pixel = np.concatenate([
                keypoints_2d,
                (keypoints_3d[..., 2] + gt_instances.root_z)[..., None]
            ],
                                        axis=-1)
            kpts_cam = kpts_pixel.copy()
            kpts_cam[..., :2] = (kpts_pixel[..., :2] - c) / f * kpts_pixel[...,
                                                                           2:]

            if mode == '3d':
                # Evaluation with 3D keypoint coordinates
                pred_instances.keypoints = kpts_cam
                pred_instances.transformed_keypoints = keypoints_2d
            elif mode == 'vis':
                # Visualization with SimCC keypoints
                pred_instances.keypoints = keypoints_simcc
                pred_instances.transformed_keypoints = keypoints_2d
            else:
                # Evaluation with 2D keypoint coordinates
                pred_instances.keypoints = keypoints_2d
                pred_instances.transformed_keypoints = keypoints_2d

            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
