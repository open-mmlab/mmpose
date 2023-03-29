# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.utils import print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.separate_eval = separate_eval

    def clip_by_index(self, results, start_idx, end_idx):
        clip_res = dict()
        clip_res['preds'] = results[0]['preds'][start_idx:end_idx, :, :]
        clip_res['boxes'] = results[0]['boxes'][start_idx:end_idx, :]
        clip_res['image_paths'] = results[0]['image_paths'][start_idx:end_idx]
        clip_res['bbox_ids'] = results[0]['bbox_ids'][start_idx:end_idx]
        return [clip_res]

    def rebuild_results(self, results):
        new_results = list()
        new_res = results[0]
        new_results.append(new_res)
        for i in range(1, len(results)):
            res = results[i]
            new_res['preds'] = np.append(new_res['preds'], res['preds'], axis=0)
            new_res['boxes'] = np.append(new_res['boxes'], res['boxes'], axis=0)
            new_res['image_paths'] += res['image_paths']
            new_res['bbox_ids'] += res['bbox_ids']
        return new_results, len(new_res['image_paths'])

    def update_total_eval_results_ap(self, eval_ap_list, total_eval_results):
        sum_num = 0
        sum_precision = 0
        for eval_ap in eval_ap_list:
            numbers = eval_ap['numbers']
            precisions = numbers * eval_ap['AP']
            sum_num += numbers
            sum_precision += precisions
        total_eval_results.update({'AP': sum_precision/sum_num})


    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """

        new_results, res_len = self.rebuild_results(results)
        assert res_len == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            total_eval_results = dict()
            eval_ap_list = list()
            for dataset_idx, dataset in enumerate(self.datasets):
                start_idx = 0 if dataset_idx == 0 else \
                    self.cumulative_sizes[dataset_idx - 1]
                end_idx = self.cumulative_sizes[dataset_idx]

                results_per_dataset = self.clip_by_index(new_results, start_idx, end_idx)
                print_log(
                    f'Evaluateing dataset-{dataset_idx} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

                if 'AP' in eval_results_per_dataset.keys():
                    eval_ap_list.append(dict(numbers=end_idx-start_idx, AP=eval_results_per_dataset['AP']))

            if len(eval_ap_list) > 0:
                self.update_total_eval_results_ap(eval_ap_list, total_eval_results)
            return total_eval_results
        elif len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types')
        else:
            raise NotImplementedError(
                'separate_eval=False not implemented')


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get data."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len
