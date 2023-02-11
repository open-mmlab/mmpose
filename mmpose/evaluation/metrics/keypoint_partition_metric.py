# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmpose.registry import METRICS


@METRICS.register_module()
class KeypointPartitionMetric(BaseMetric):
    """Wrapper metric for evaluating pose metric on user-defined body parts.

    Sometimes one may be interested in the performance of a pose model on
    certain body parts rather than on all the keypoints. For example,
    ``CocoWholeBodyMetric`` evaluates coco metric on body, foot, face,
    lefthand and righthand. However, ``CocoWholeBodyMetric`` cannot be
    applied to arbitrary custom datasets. This wrapper metric solves this
    problem.

    Supported metrics:
        ``CocoMetric``  Note 1: all keypoint ground truth should be stored in
            `keypoints` not other data fields. Note 2: `ann_file` is not
            supported, it will be ignored. Note 3: `score_mode` other than
            'bbox' may produce results different from the
            ``CocoWholebodyMetric``. Note 4: `nms_mode` other than 'none' may
            produce results different from the ``CocoWholebodyMetric``.
        ``PCKAccuracy`` Note 1: data fields required by ``PCKAccuracy`` should
         be provided, such as bbox, head_size, etc. Note 2: In terms of
        'torso', since it is specifically designed for ``JhmdbDataset``, it is
         not recommended to use it for other datasets.
        ``AUC`` supported without limitations.
        ``EPE`` supported without limitations.
        ``NME`` only `norm_mode` = 'use_norm_item' is supported,
        'keypoint_distance' is incompatible with ``KeypointPartitionMetric``.

    Incompatible metrics:
        The following metrics are dataset specific metrics:
            ``CocoWholeBodyMetric``
            ``MpiiPCKAccuracy``
            ``JhmdbPCKAccuracy``
            ``PoseTrack18Metric``
        Keypoint partitioning is included in these metrics.

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
        # check metric type
        supported_metric_types = [
            'CocoMetric', 'PCKAccuracy', 'AUC', 'EPE', 'NME'
        ]
        if metric['type'] not in supported_metric_types:
            raise ValueError(
                'Metrics supported by KeypointPartitionMetric are CocoMetric, '
                'PCKAccuracy, AUC, EPE and NME, '
                f"but got {metric['type']}")

        # check CocoMetric arguments
        if metric['type'] == 'CocoMetric':
            if 'ann_file' in metric:
                warnings.warn(
                    'KeypointPartitionMetric does not support the ann_file '
                    'argument of CocoMetric, this argument will be ignored.')
                metric['ann_file'] = None
            score_mode = metric.get('score_mode', 'bbox_keypoint')
            if score_mode != 'bbox':
                warnings.warn(
                    'When using KeypointPartitionMetric with CocoMetric, '
                    "if score_mode is not 'bbox', pose scores will be "
                    "calculated part by part rather than by 'wholebody'. "
                    'Therefore, this may produce results different from the '
                    'CocoWholebodyMetric.')
            nms_mode = metric.get('nms_mode', 'oks_nms')
            if nms_mode != 'none':
                warnings.warn(
                    'When using KeypointPartitionMetric with CocoMetric, '
                    'oks_nms and soft_oks_nms will be calculated part by part '
                    "rather than by 'wholebody'. Therefore, this may produce "
                    'results different from the CocoWholebodyMetric.')

        # check PCKAccuracy arguments
        if metric['type'] == 'PCKAccuracy':
            norm_item = metric.get('norm_item', 'bbox')
            if norm_item == 'torso' or 'torso' in norm_item:
                warnings.warn(
                    'norm_item torso is used in JhmdbDataset, it may not be '
                    'compatible with other datasets, use at your own risk.')

        # check NME arguments
        if metric['type'] == 'NME':
            assert 'norm_mode' in metric, \
                'Missing norm_mode required by the NME metric.'
            if metric['norm_mode'] != 'use_norm_item':
                raise ValueError(
                    "NME norm_mode 'keypoint_distance' is incompatible with "
                    'KeypointPartitionMetric.')

        # check partitions
        assert len(partitions) > 0, 'There should be at least one partition.'
        for partition_name, partition in partitions.items():
            assert isinstance(partition, Sequence), \
                'Each partition should be a sequence.'
            assert len(partition) > 0, \
                'Each partition should have at least one element.'
        self.partitions = partitions

        # instantiate metrics for each partition
        self.metrics = {}
        for partition_name in partitions.keys():
            _metric = deepcopy(metric)
            if 'outfile_prefix' in _metric:
                _metric['outfile_prefix'] = _metric[
                    'outfile_prefix'] + '.' + partition_name
            self.metrics[partition_name] = METRICS.build(_metric)

    @BaseMetric.dataset_meta.setter
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
                if 'keypoint_scores' in _data_sample['pred_instances']:
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
                    raw_ann_info = _data_sample['raw_ann_info']
                    anns = raw_ann_info if isinstance(
                        raw_ann_info, list) else [raw_ann_info]
                    for ann in anns:
                        if 'keypoints' in ann:
                            keypoints = np.array(ann['keypoints']).reshape(
                                -1, 3)
                            keypoints = keypoints[keypoint_ids]
                            num_keypoints = np.sum(keypoints[:, 2] > 0)
                            ann['keypoints'] = keypoints.flatten().tolist()
                            ann['num_keypoints'] = num_keypoints

                parted_data_samples[partition_name].append(_data_sample)

        for partition_name, metric in self.metrics.items():
            metric.process(data_batch, parted_data_samples[partition_name])

    def compute_metrics(self, results: list) -> dict:
        pass

    def evaluate(self, size: int) -> dict:
        """Run evaluation for each partition."""
        eval_results = OrderedDict()
        for partition_name, metric in self.metrics.items():
            _eval_results = metric.evaluate(size)
            for key in list(_eval_results.keys()):
                new_key = partition_name + '/' + key
                _eval_results[new_key] = _eval_results.pop(key)
            eval_results.update(_eval_results)
        return eval_results
