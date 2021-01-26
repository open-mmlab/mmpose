import os.path as osp
from math import inf

import mmcv
from mmcv.runner import Hook
from torch.utils.data import DataLoader

from mmpose.utils import get_root_logger


class EvalHook(Hook):
    """Non-Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs).
            Default: 1.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        save_best (bool): Whether to save best checkpoint during evaluation.
            Default: True.
        key_indicator (str | None): Key indicator to measure the best
            checkpoint during evaluation when ``save_best`` is set to True.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``acc``, ``AP``, ``PCK``. Default: `AP`.
        rule (str | None): Comparison rule for best score. If set to None,
            it will infer a reasonable rule. Default: 'None'.
        eval_kwargs (dict, optional): Arguments for evaluation.
    """

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = ['acc', 'ap', 'ar', 'pck', 'auc']
    less_keys = ['loss', 'epe', 'nme']

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 save_best=True,
                 key_indicator='AP',
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')
        if save_best and not key_indicator:
            raise ValueError('key_indicator should not be None, when '
                             'save_best is set to True.')
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None and save_best:
            if any(key in key_indicator.lower() for key in self.greater_keys):
                rule = 'greater'
            elif any(key in key_indicator.lower() for key in self.less_keys):
                rule = 'less'
            else:
                raise ValueError(
                    f'key_indicator must be in {self.greater_keys} '
                    f'or in {self.less_keys} when rule is None, '
                    f'but got {key_indicator}')

        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs
        self.save_best = save_best
        self.key_indicator = key_indicator
        self.rule = rule

        self.logger = get_root_logger()

        if self.save_best:
            self.compare_func = self.rule_map[self.rule]
            self.best_score = self.init_value_map[self.rule]

        self.best_json = dict()

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if not self.every_n_epochs(runner, self.interval):
            return

        current_ckpt_path = osp.join(runner.work_dir,
                                     f'epoch_{runner.epoch + 1}.pth')
        json_path = osp.join(runner.work_dir, 'best.json')

        if osp.exists(json_path) and len(self.best_json) == 0:
            self.best_json = mmcv.load(json_path)
            self.best_score = self.best_json['best_score']
            self.best_ckpt = self.best_json['best_ckpt']
            self.key_indicator = self.best_json['key_indicator']

        from mmpose.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader)
        key_score = self.evaluate(runner, results)
        if (self.save_best and self.compare_func(key_score, self.best_score)):
            self.best_score = key_score
            self.logger.info(
                f'Now best checkpoint is epoch_{runner.epoch + 1}.pth')
            self.best_json['best_score'] = self.best_score
            self.best_json['best_ckpt'] = current_ckpt_path
            self.best_json['key_indicator'] = self.key_indicator
            mmcv.dump(self.best_json, json_path)

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, runner.work_dir, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        if self.key_indicator is not None:
            return eval_res[self.key_indicator]

        return None


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        save_best (bool): Whether to save best checkpoint during evaluation.
            Default: True.
        key_indicator (str | None): Key indicator to measure the best
            checkpoint during evaluation when ``save_best`` is set to True.
            Options are the evaluation metrics to the test dataset. e.g.,
            ``top1_acc``, ``top5_acc``, ``mean_class_accuracy``,
            ``mean_average_precision`` for action recognition dataset
            (RawframeDataset and VideoDataset). ``AR@AN``, ``auc`` for action
            localization dataset (ActivityNetDataset). Default: `top1_acc`.
        rule (str | None): Comparison rule for best score. If set to None,
            it will infer a reasonable rule. Default: 'None'.
        eval_kwargs (dict, optional): Arguments for evaluation.
    """

    def after_train_epoch(self, runner):
        """Called after each training epoch to evaluate the model."""
        if not self.every_n_epochs(runner, self.interval):
            return

        current_ckpt_path = osp.join(runner.work_dir,
                                     f'epoch_{runner.epoch + 1}.pth')
        json_path = osp.join(runner.work_dir, 'best.json')

        if osp.exists(json_path) and len(self.best_json) == 0:
            self.best_json = mmcv.load(json_path)
            self.best_score = self.best_json['best_score']
            self.best_ckpt = self.best_json['best_ckpt']
            self.key_indicator = self.best_json['key_indicator']

        from mmpose.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if (self.save_best
                    and self.compare_func(key_score, self.best_score)):
                self.best_score = key_score
                self.logger.info(
                    f'Now best checkpoint is epoch_{runner.epoch + 1}.pth')
                self.best_json['best_score'] = self.best_score
                self.best_json['best_ckpt'] = current_ckpt_path
                self.best_json['key_indicator'] = self.key_indicator
                mmcv.dump(self.best_json, json_path)
