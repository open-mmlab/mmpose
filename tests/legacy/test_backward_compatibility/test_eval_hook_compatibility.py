# Copyright (c) OpenMMLab. All rights reserved.
import unittest.mock as mock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mmpose.core import DistEvalHook, EvalHook


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.eval_result = [0.1, 0.4, 0.3, 0.7, 0.2, 0.05, 0.4, 0.6]

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 1

    @mock.create_autospec
    def evaluate(self, results, res_folder=None, logger=None):
        pass


def test_old_fashion_eval_hook_parameters():

    data_loader = DataLoader(
        ExampleDataset(),
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=False)

    # test argument "key_indicator"
    with pytest.warns(DeprecationWarning):
        _ = EvalHook(data_loader, key_indicator='AP')
    with pytest.warns(DeprecationWarning):
        _ = DistEvalHook(data_loader, key_indicator='AP')

    # test argument "gpu_collect"
    with pytest.warns(DeprecationWarning):
        _ = EvalHook(data_loader, save_best='AP', gpu_collect=False)
