# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from torch.utils.data import Dataset

from mmpose.engine.hooks import RTMOModeSwitchHook, YOLOXPoseModeSwitchHook
from mmpose.utils import register_all_modules


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


pipeline1 = [
    dict(type='RandomHalfBody'),
]

pipeline2 = [
    dict(type='RandomFlip'),
]
register_all_modules()


class TestYOLOXPoseModeSwitchHook(TestCase):

    def test(self):
        train_dataloader = dict(
            dataset=DummyDataset(),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=3,
            num_workers=0)

        runner = Mock()
        runner.model = Mock()
        runner.model.module = Mock()

        runner.model.head.use_aux_loss = False
        runner.cfg.train_dataloader = Config(train_dataloader)
        runner.train_dataloader = Runner.build_dataloader(train_dataloader)
        runner.train_dataloader.dataset.pipeline = pipeline1

        hook = YOLOXPoseModeSwitchHook(
            num_last_epochs=15, new_train_pipeline=pipeline2)

        # test after change mode
        runner.epoch = 284
        runner.max_epochs = 300
        hook.before_train_epoch(runner)
        self.assertTrue(runner.model.bbox_head.use_aux_loss)
        self.assertEqual(runner.train_loop.dataloader.dataset.pipeline,
                         pipeline2)


class TestRTMOModeSwitchHook(TestCase):

    def test(self):

        runner = Mock()
        runner.model = Mock()
        runner.model.head = Mock()
        runner.model.head.loss = Mock()

        runner.model.head.attr1 = False
        runner.model.head.loss.attr2 = 1.0

        hook = RTMOModeSwitchHook(epoch_attributes={
            0: {
                'attr1': True
            },
            10: {
                'loss.attr2': 0.5
            }
        })

        # test after change mode
        runner.epoch = 0
        hook.before_train_epoch(runner)
        self.assertTrue(runner.model.head.attr1)
        self.assertEqual(runner.model.head.loss.attr2, 1.0)

        runner.epoch = 10
        hook.before_train_epoch(runner)
        self.assertTrue(runner.model.head.attr1)
        self.assertEqual(runner.model.head.loss.attr2, 0.5)
