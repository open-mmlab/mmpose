# Copyright (c) OpenMMLab. All rights reserved.

from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmpose.registry import METRICS


@METRICS.register_module()
class KeypointPartitionMetric(BaseMetric):
    """Wrapper metric for evaluating pose metric on user-defined body parts.

    Sometimes one may be interested in the performance of a pose model on
    certain body parts rather than on all the keypoints. For example,
    ``CocoWholeBodyMetric`` evaluates coco metric on body, foot, face,
    lefthand and righthand. However, ``CocoWholeBodyMetric`` is not flexible
    enough to be applied to arbitrary custom datasets. Therefore, we provide
    this wrapper metric.

    Supported metrics:
        ``CocoMetric``  all keypoint ground truth should be stored in
            `keypoints` not other data fields.
        ``PCKAccuracy`` the required data fields should be provided in
            annotation files, such as head bbox, etc.
        ``AUC``
        ``EPE``

    Args:
        metric (dict): arguments to instantiate a metric, please refer to the
            arguments required by the metric of your choice.
        partitions (dict): definition of body partitions. For example, if we
            have 10 keypoints in total, the first 7 keypoints belong to body
            and the last 3 keypoints belong to foot, this field can be like
            this:
                dict(
                    body=[0, 1, 2, 3, 4, 5, 6],
                    foot=[7, 8, 9],
                    all=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                )
            where the numbers are the indices of keypoints and they can be
            discontinuous.
    """

    def __init__(
        self,
        metric: dict,
        partitions: dict,
    ) -> None:
        super().__init__()
        self.partitions = partitions
        # instantiate metrics for each partition
        self.metrics = {
            partition_name: METRICS.build(metric)
            for partition_name in partitions.keys()
        }

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta
        # sigmas required by coco metric have to be split as well
        for partition_name, keypoint_ids in self.partitions.items():
            _dataset_meta = deepcopy(dataset_meta)
            _dataset_meta['num_keypoints'] = len(keypoint_ids)
            _dataset_meta['sigmas'] = _dataset_meta['sigmas'][keypoint_ids]
            self.metrics[partition_name].dataset_meta = _dataset_meta

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Split data samples by partitions, then call metric.process part by
        part."""
        parted_data_samples = {
            partition_name: []
            for partition_name in self.partitions.keys()
        }
        for data_sample in data_samples:
            for partition_name, keypoint_ids in self.partitions.items():
                _data_sample = deepcopy(data_sample)
                _data_sample['pred_instances'][
                    'keypoint_scores'] = _data_sample['pred_instances'][
                        'keypoint_scores'][:, keypoint_ids]
                _data_sample['pred_instances']['keypoints'] = _data_sample[
                    'pred_instances']['keypoints'][:, keypoint_ids]
                _data_sample['gt_instances']['keypoints'] = _data_sample[
                    'gt_instances']['keypoints'][:, keypoint_ids]
                _data_sample['gt_instances'][
                    'keypoints_visible'] = _data_sample['gt_instances'][
                        'keypoints_visible'][:, keypoint_ids]

                # for coco metric
                if 'raw_ann_info' in _data_sample:
                    if 'keypoints' in _data_sample['raw_ann_info']:
                        keypoints = np.array(
                            _data_sample['raw_ann_info']['keypoints']).reshape(
                                -1, 3)
                        keypoints = keypoints[keypoint_ids]
                        num_keypoints = np.sum(keypoints[:, 2] > 0)
                        _data_sample['raw_ann_info'][
                            'keypoints'] = keypoints.flatten().tolist()
                        _data_sample['raw_ann_info'][
                            'num_keypoints'] = num_keypoints

                parted_data_samples[partition_name].append(_data_sample)

        for partition_name, metric in self.metrics.items():
            metric.process(data_batch, parted_data_samples[partition_name])

    def compute_metrics(self, results: list) -> dict:
        pass

    def evaluate(self, size: int) -> dict:
        """Overload the evaluate method of the base class.

        Run evaluation for each partitions one by one.
        """
        eval_results = OrderedDict()
        for partition_name, metric in self.metrics.items():
            _eval_results = metric.evaluate(size)
            for key in list(_eval_results.keys()):
                new_key = partition_name + '/' + key
                _eval_results[new_key] = _eval_results.pop(key)
            eval_results.update(_eval_results)
        return eval_results
