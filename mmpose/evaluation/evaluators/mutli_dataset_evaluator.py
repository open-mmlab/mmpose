# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Any, Optional, Sequence, Union

from mmengine.evaluator.evaluator import Evaluator
from mmengine.evaluator.metric import BaseMetric
from mmengine.structures import BaseDataElement

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import DATASETS, EVALUATORS


@EVALUATORS.register_module()
class MultiDatasetEvaluator(Evaluator):
    """Wrapper class to compose multiple :class:`BaseMetric` instances.

    Args:
        metrics (dict or BaseMetric or Sequence): The configs of metrics.
        datasets (Sequence[str]): The configs of datasets.
    """

    def __init__(
        self,
        metrics: Union[dict, BaseMetric, Sequence],
        datasets: Sequence[dict],
    ):

        assert len(metrics) == len(datasets), 'the argument ' \
            'datasets should have same length as metrics'

        super().__init__(metrics)

        # Initialize metrics for each dataset
        metrics_dict = dict()
        for dataset, metric in zip(datasets, self.metrics):
            metainfo_file = DATASETS.module_dict[dataset['type']].METAINFO
            dataset_meta = parse_pose_metainfo(metainfo_file)
            metric.dataset_meta = dataset_meta
            metrics_dict[dataset_meta['dataset_name']] = metric
        self.metrics_dict = metrics_dict

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the evaluator and it's metrics."""
        self._dataset_meta = dataset_meta

    def process(self,
                data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """
        _data_samples = defaultdict(list)
        _data_batch = dict(
            inputs=defaultdict(list),
            data_samples=defaultdict(list),
        )

        for inputs, data_ds, data_sample in zip(data_batch['inputs'],
                                                data_batch['data_samples'],
                                                data_samples):
            if isinstance(data_sample, BaseDataElement):
                data_sample = data_sample.to_dict()
            assert isinstance(data_sample, dict)
            dataset_name = data_sample.get('dataset_name',
                                           self.dataset_meta['dataset_name'])
            _data_samples[dataset_name].append(data_sample)
            _data_batch['inputs'][dataset_name].append(inputs)
            _data_batch['data_samples'][dataset_name].append(data_ds)

        for dataset_name, metric in self.metrics_dict.items():
            if dataset_name in _data_samples:
                data_batch = dict(
                    inputs=_data_batch['inputs'][dataset_name],
                    data_samples=_data_batch['data_samples'][dataset_name])
                metric.process(data_batch, _data_samples[dataset_name])
            else:
                continue
