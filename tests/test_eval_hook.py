import os.path as osp
import tempfile
import unittest.mock as mock

import mmcv
import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmpose.core import EvalHook


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.eval_result = [1, 4, 3, 7, 2, -3, 4, 6]

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 1

    @mock.create_autospec
    def evaluate(self, results, work_dir, logger=None):
        pass


class EvalDataset(ExampleDataset):

    def evaluate(self, results, work_dir, logger=None):
        acc = self.eval_result[self.index]
        output = dict(acc=acc, index=self.index, score=acc)
        self.index += 1
        return output


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return imgs

    def train_step(self, data_batch, optimizer, **kwargs):
        outputs = {
            'loss': 0.5,
            'log_vars': {
                'accuracy': 0.98
            },
            'num_samples': 1
        }
        return outputs


def test_eval_hook():
    with pytest.raises(TypeError):
        # dataloader must be a pytorch DataLoader
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        EvalHook(data_loader)

    with pytest.raises(ValueError):
        # when `save_best` is True, `key_indicator` should not be None
        test_dataset = ExampleModel()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        EvalHook(data_loader, key_indicator=None)

    with pytest.raises(KeyError):
        # rule must be in keys of rule_map
        test_dataset = ExampleModel()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        EvalHook(data_loader, save_best=False, rule='unsupport')

    with pytest.raises(ValueError):
        # key_indicator must be valid when rule_map is None
        test_dataset = ExampleModel()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        EvalHook(data_loader, key_indicator='unsupport')

    optimizer_cfg = dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)

    data_loader = DataLoader(test_dataset, batch_size=1)
    eval_hook = EvalHook(data_loader, save_best=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)

        best_json_path = osp.join(tmpdir, 'best.json')
        assert not osp.exists(best_json_path)

    loader = DataLoader(EvalDataset(), batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(
        data_loader, interval=1, save_best=True, key_indicator='acc')

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_4.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 7
        assert best_json['key_indicator'] == 'acc'

    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(
        data_loader,
        interval=1,
        save_best=True,
        key_indicator='score',
        rule='greater')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_4.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 7
        assert best_json['key_indicator'] == 'score'

    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(data_loader, rule='less', key_indicator='acc')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_6.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == -3
        assert best_json['key_indicator'] == 'acc'

    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(data_loader, key_indicator='acc')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 2)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_2.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 4
        assert best_json['key_indicator'] == 'acc'

        resume_from = osp.join(tmpdir, 'latest.pth')
        loader = DataLoader(ExampleDataset(), batch_size=1)
        eval_hook = EvalHook(data_loader, key_indicator='acc')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.resume(resume_from)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_4.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 7
        assert best_json['key_indicator'] == 'acc'
