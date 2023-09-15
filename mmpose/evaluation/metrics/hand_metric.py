# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.codecs.utils import pixel_to_camera
from mmpose.registry import METRICS
from ..functional import keypoint_epe


@METRICS.register_module()
class InterHandMetric(BaseMetric):

    METRICS = {'MPJPE', 'MRRPE', 'HandednessAcc'}

    def __init__(self,
                 modes: List[str] = ['MPJPE', 'MRRPE', 'HandednessAcc'],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        for mode in modes:
            if mode not in self.METRICS:
                raise ValueError("`mode` should be 'MPJPE', 'MRRPE', or "
                                 f"'HandednessAcc', but got '{mode}'.")

        self.modes = modes

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            _, K, _ = pred_coords.shape
            pred_coords_cam = pred_coords.copy()
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints_cam']

            keypoints_cam = gt_coords.copy()
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool).reshape(1, -1)

            pred_hand_type = data_sample['pred_instances']['hand_type']
            gt_hand_type = data_sample['hand_type']
            if pred_hand_type is None and 'HandednessAcc' in self.modes:
                raise KeyError('metric HandednessAcc is not supported')

            pred_root_depth = data_sample['pred_instances']['rel_root_depth']
            if pred_root_depth is None and 'MRRPE' in self.modes:
                raise KeyError('metric MRRPE is not supported')

            abs_depth = data_sample['abs_depth']
            focal = data_sample['focal']
            principal_pt = data_sample['principal_pt']

            result = {}

            if 'MPJPE' in self.modes:
                keypoints_cam[..., :21, :] -= keypoints_cam[..., 20, :]
                keypoints_cam[..., 21:, :] -= keypoints_cam[..., 41, :]

                pred_coords_cam[..., :21, 2] += abs_depth[0]
                pred_coords_cam[..., 21:, 2] += abs_depth[1]
                pred_coords_cam = pixel_to_camera(pred_coords_cam, focal[0],
                                                  focal[1], principal_pt[0],
                                                  principal_pt[1])

                pred_coords_cam[..., :21, :] -= pred_coords_cam[..., 20, :]
                pred_coords_cam[..., 21:, :] -= pred_coords_cam[..., 41, :]

                if gt_hand_type.all():
                    single_mask = np.zeros((1, K), dtype=bool)
                    interacting_mask = mask
                else:
                    single_mask = mask
                    interacting_mask = np.zeros((1, K), dtype=bool)

                result['pred_coords'] = pred_coords_cam
                result['gt_coords'] = keypoints_cam
                result['mask'] = mask
                result['single_mask'] = single_mask
                result['interacting_mask'] = interacting_mask

            if 'HandednessAcc' in self.modes:
                hand_type_mask = data_sample['hand_type_valid'] > 0
                result['pred_hand_type'] = pred_hand_type
                result['gt_hand_type'] = gt_hand_type
                result['hand_type_mask'] = hand_type_mask

            if 'MRRPE' in self.modes:
                keypoints_visible = gt['keypoints_visible']
                if gt_hand_type.all() and keypoints_visible[
                        ..., 20] and keypoints_visible[..., 41]:
                    rel_root_mask = np.array([True])

                    pred_left_root_coords = np.array(
                        pred_coords[..., 41, :], dtype=np.float32)
                    pred_left_root_coords[...,
                                          2] += abs_depth[0] + pred_root_depth
                    pred_left_root_coords = pixel_to_camera(
                        pred_left_root_coords, focal[0], focal[1],
                        principal_pt[0], principal_pt[1])

                    pred_right_root_coords = np.array(
                        pred_coords[..., 20, :], dtype=np.float32)
                    pred_right_root_coords[..., 2] += abs_depth[0]
                    pred_right_root_coords = pixel_to_camera(
                        pred_right_root_coords, focal[0], focal[1],
                        principal_pt[0], principal_pt[1])
                    pred_rel_root_coords = pred_left_root_coords - \
                        pred_right_root_coords
                    pred_rel_root_coords = np.expand_dims(
                        pred_rel_root_coords, axis=0)
                    gt_rel_root_coords = gt_coords[...,
                                                   41, :] - gt_coords[...,
                                                                      20, :]
                    gt_rel_root_coords = np.expand_dims(
                        gt_rel_root_coords, axis=0)
                else:
                    rel_root_mask = np.array([False])
                    pred_rel_root_coords = np.array([[0, 0, 0]])
                    pred_rel_root_coords = pred_rel_root_coords.reshape(
                        1, 1, 3)
                    gt_rel_root_coords = np.array([[0, 0, 0]]).reshape(1, 1, 3)

                result['pred_rel_root_coords'] = pred_rel_root_coords
                result['gt_rel_root_coords'] = gt_rel_root_coords
                result['rel_root_mask'] = rel_root_mask

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        metrics = dict()

        logger.info(f'Evaluating {self.__class__.__name__}...')

        if 'MPJPE' in self.modes:
            # pred_coords: [N, K, D]
            pred_coords = np.concatenate(
                [result['pred_coords'] for result in results])
            # gt_coords: [N, K, D]
            gt_coords = np.concatenate(
                [result['gt_coords'] for result in results])
            # mask: [N, K]
            mask = np.concatenate([result['mask'] for result in results])
            single_mask = np.concatenate(
                [result['single_mask'] for result in results])
            interacting_mask = np.concatenate(
                [result['interacting_mask'] for result in results])

            metrics['MPJPE_all'] = keypoint_epe(pred_coords, gt_coords, mask)
            metrics['MPJPE_single'] = keypoint_epe(pred_coords, gt_coords,
                                                   single_mask)
            metrics['MPJPE_interacting'] = keypoint_epe(
                pred_coords, gt_coords, interacting_mask)

        if 'HandednessAcc' in self.modes:
            pred_hand_type = np.concatenate(
                [result['pred_hand_type'] for result in results])
            gt_hand_type = np.concatenate(
                [result['gt_hand_type'] for result in results])
            hand_type_mask = np.concatenate(
                [result['hand_type_mask'] for result in results])
            acc = (pred_hand_type == gt_hand_type).all(axis=-1)
            metrics['HandednessAcc'] = np.mean(acc[hand_type_mask])

        if 'MRRPE' in self.modes:
            pred_rel_root_coords = np.concatenate(
                [result['pred_rel_root_coords'] for result in results])
            gt_rel_root_coords = np.concatenate(
                [result['gt_rel_root_coords'] for result in results])
            rel_root_mask = np.array(
                [result['rel_root_mask'] for result in results])
            metrics['MRRPE'] = keypoint_epe(pred_rel_root_coords,
                                            gt_rel_root_coords, rel_root_mask)
        return metrics
