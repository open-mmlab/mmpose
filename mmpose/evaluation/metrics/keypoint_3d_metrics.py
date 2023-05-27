# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from os import path as osp
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import keypoint_mpjpe


@METRICS.register_module()
class MPJPE(BaseMetric):
    """MPJPE evaluation metric.

    Calculate the mean per-joint position error (MPJPE) of keypoints.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        mode (str): Method to align the prediction with the
            ground truth. Supported options are:

                - ``'mpjpe'``: no alignment will be applied
                - ``'p-mpjpe'``: align in the least-square sense in scale
                - ``'n-mpjpe'``: align in the least-square sense in
                    scale, rotation, and translation.

        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    ALIGNMENT = {'mpjpe': 'none', 'p-mpjpe': 'procrustes', 'n-mpjpe': 'scale'}

    def __init__(self,
                 mode: str = 'mpjpe',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        allowed_modes = self.ALIGNMENT.keys()
        if mode not in allowed_modes:
            raise KeyError("`mode` should be 'mpjpe', 'p-mpjpe', or "
                           f"'n-mpjpe', but got '{mode}'.")

        self.mode = mode

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
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['lifting_target']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['lifting_target_visible'].astype(bool).reshape(1, -1)
            # instance action
            img_path = data_sample['target_img_path']
            _, rest = osp.basename(img_path).split('_', 1)
            action, _ = rest.split('.', 1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
                'action': action
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are the corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        if pred_coords.ndim == 4 and pred_coords.shape[1] == 1:
            pred_coords = np.squeeze(pred_coords, axis=1)
        # gt_coords: [N, K, D]
        gt_coords = np.stack([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])
        # action_category_indices: Dict[List[int]]
        action_category_indices = defaultdict(list)
        for idx, result in enumerate(results):
            action_category = result['action'].split('_')[0]
            action_category_indices[action_category].append(idx)

        error_name = self.mode.upper()

        logger.info(f'Evaluating {self.mode.upper()}...')
        metrics = dict()

        metrics[error_name] = keypoint_mpjpe(pred_coords, gt_coords, mask,
                                             self.ALIGNMENT[self.mode])

        for action_category, indices in action_category_indices.items():
            metrics[f'{error_name}_{action_category}'] = keypoint_mpjpe(
                pred_coords[indices], gt_coords[indices], mask[indices])

        return metrics
