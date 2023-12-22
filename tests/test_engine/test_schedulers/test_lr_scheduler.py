# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F
import torch.optim as optim
from mmengine.optim.scheduler import _ParamScheduler
from mmengine.testing import assert_allclose

from mmpose.engine.schedulers import ConstantLR


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestLRScheduler(TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.model = ToyModel()
        lr = 0.05
        self.layer2_mult = 10
        self.optimizer = optim.SGD([{
            'params': self.model.conv1.parameters()
        }, {
            'params': self.model.conv2.parameters(),
            'lr': lr * self.layer2_mult,
        }],
                                   lr=lr,
                                   momentum=0.01,
                                   weight_decay=5e-4)

    def _test_scheduler_value(self,
                              schedulers,
                              targets,
                              epochs=10,
                              param_name='lr',
                              step_kwargs=None):
        if isinstance(schedulers, _ParamScheduler):
            schedulers = [schedulers]
        if step_kwargs is None:
            step_kwarg = [{} for _ in range(len(schedulers))]
            step_kwargs = [step_kwarg for _ in range(epochs)]
        else:  # step_kwargs is not None
            assert len(step_kwargs) == epochs
            assert len(step_kwargs[0]) == len(schedulers)
        for epoch in range(epochs):
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                assert_allclose(
                    target[epoch],
                    param_group[param_name],
                    msg='{} is wrong in epoch {}: expected {}, got {}'.format(
                        param_name, epoch, target[epoch],
                        param_group[param_name]),
                    atol=1e-5,
                    rtol=0)
            [
                scheduler.step(**step_kwargs[epoch][i])
                for i, scheduler in enumerate(schedulers)
            ]

    def test_constant_scheduler(self):

        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 4 + [0.05] * 6
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler = ConstantLR(self.optimizer, factor=1.0 / 2, end=5)
        self._test_scheduler_value(scheduler, targets, epochs)

        # remove factor range restriction
        _ = ConstantLR(self.optimizer, factor=99, end=100)
