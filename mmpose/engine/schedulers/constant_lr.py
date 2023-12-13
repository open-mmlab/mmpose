# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.scheduler import \
    ConstantParamScheduler as MMENGINE_ConstantParamScheduler
from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin

from mmpose.registry import PARAM_SCHEDULERS

INF = int(1e9)


class ConstantParamScheduler(MMENGINE_ConstantParamScheduler):
    """Decays the parameter value of each parameter group by a small constant
    factor until the number of epoch reaches a pre-defined milestone: ``end``.
    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler. The factor range restriction
    is removed.

    Args:
        optimizer (Optimizer or BaseOptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        factor (float): The number we multiply parameter value until the
            milestone. Defaults to 1./3.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer,
                 param_name: str,
                 factor: float = 1.0 / 3,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        self.factor = factor
        self.total_iters = end - begin - 1
        super(MMENGINE_ConstantParamScheduler, self).__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@PARAM_SCHEDULERS.register_module()
class ConstantLR(LRSchedulerMixin, ConstantParamScheduler):
    """Decays the learning rate value of each parameter group by a small
    constant factor until the number of epoch reaches a pre-defined milestone:
    ``end``. Notice that such decay can happen simultaneously with other
    changes to the learning rate value from outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the
            milestone. Defaults to 1./3.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without state
            dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
    """
