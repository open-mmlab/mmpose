
from collections import OrderedDict
from typing import Sequence, Dict, Optional
import numpy as np

from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS
from copy import deepcopy


@METRICS.register_module()
class PartitionMetric(BaseMetric):
    def __init__(self,
                metric: dict,
                partitions: dict,
                collect_device: str = 'cpu',
                ) -> None:
        super().__init__(collect_device=collect_device)
        self.partitions = partitions
        self.metrics = {}
        _prefix = metric.get('prefix', None)
        for partition_name in partitions.keys():
            metric['prefix'] = _prefix + '/' + partition_name if _prefix is not None else partition_name
            self.metrics[partition_name] = METRICS.build(metric)

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta
        for partition_name, keypoint_ids in self.partitions.items():
            _dataset_meta = deepcopy(dataset_meta)
            _dataset_meta['num_keypoints'] = len(keypoint_ids)
            _dataset_meta['sigmas'] = _dataset_meta['sigmas'][keypoint_ids]
            self.metrics[partition_name].dataset_meta = _dataset_meta

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        parted_data_samples = {partition_name: [] for partition_name in self.partitions.keys()}
        for data_sample in data_samples:
            for partition_name, keypoint_ids in self.partitions.items():
                _data_sample = deepcopy(data_sample)
                _data_sample['pred_instances']['keypoint_scores'] = _data_sample['pred_instances']['keypoint_scores'][:, keypoint_ids]
                _data_sample['pred_instances']['keypoints'] = _data_sample['pred_instances']['keypoints'][:, keypoint_ids]
                _data_sample['gt_instances']['keypoints'] = _data_sample['gt_instances']['keypoints'][:, keypoint_ids]
                _data_sample['gt_instances']['keypoints_visible'] = _data_sample['gt_instances']['keypoints_visible'][:, keypoint_ids]

                # coco
                if 'raw_ann_info' in _data_sample:
                    if 'keypoints' in _data_sample['raw_ann_info']:
                        keypoints = np.array(_data_sample['raw_ann_info']['keypoints']).reshape(-1, 3)
                        keypoints = keypoints[keypoint_ids]
                        num_keypoints = np.sum(keypoints[:, 2] > 0)
                        _data_sample['raw_ann_info']['keypoints'] = keypoints.flatten().tolist()
                        _data_sample['raw_ann_info']['num_keypoints'] = num_keypoints

                parted_data_samples[partition_name].append(_data_sample)
        
        for partition_name, metric in self.metrics.items():
            metric.process(data_batch, parted_data_samples[partition_name])

    def compute_metrics(self, results: list) -> dict:
        pass

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.
        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.
        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """

        eval_results = OrderedDict()
        for partition_name, metric in self.metrics.items():
            _eval_results = metric.evaluate(size)
            eval_results.update(_eval_results)
        return eval_results
