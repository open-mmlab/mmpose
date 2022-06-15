# 教程 6: 自定义运行配置

在这篇教程中，我们将会介绍如何在您的项目中自定义优化方法、训练策略、工作流和钩子。

<!-- TOC -->

- [自定义优化方法](#自定义优化方法)
  - [使用PyTorch支持的优化器](#使用PyTorch支持的优化器)
  - [使用自己实现的优化器](#使用自己实现的优化器)
    - [1. 定义一个新优化器](#1-定义一个新优化器)
    - [2. 注册这个优化器](#2-注册这个优化器)
    - [3. 在配置文件中指定优化器](#3-在配置文件中指定优化器)
  - [自定义优化器构造器](#自定义优化器构造器)
  - [更多设置](#更多设置)
- [自定义训练策略](#自定义训练策略)
- [自定义工作流](#自定义工作流)
- [自定义钩子](#customize-hooks)
  - [使用自己实现的钩子](#customize-self-implemented-hooks)
    - [1. 定义一个新的钩子](#1-定义一个新的钩子)
    - [2. 注册这个新的钩子](#2-注册这个新的钩子)
    - [3. 修改配置文件](#3-修改配置文件)
  - [使用MMCV中的钩子](#使用MMCV中的钩子)
  - [修改默认的运行钩子](#修改默认的运行钩子)
    - [模型权重文件配置](#模型权重文件配置)
    - [日志配置](#日志配置)
    - [测试配置](#测试配置)

<!-- TOC -->

## 自定义优化方法

### 使用PyTorch支持的优化器

我们现已支持PyTorch自带的所有优化器。若要使用这些优化器，用户只需在配置文件中修改 `optimizer` 这一项。比如说，若您想使用 `Adam` 优化器，可以对配置文件做如下修改

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

若要修改模型的学习率，用户只需在配置文件中修改优化器的 `lr` 参数。优化器各参数的设置可参考PyTorch的[API文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim)。

例如，用户想要使用在PyTorch中配置为 `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` 的 `Adam` 优化器，可按照以下形式修改配置文件。

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

### 使用自己实现的优化器

#### 1. 定义一个新优化器

如果您想添加一个新的优化器，名字叫`MyOptimizer`，参数包括 `a` 、`b` 、`c`，可以按照以下步骤定义该优化器。

首先，创建一个新目录 `mmpose/core/optimizer`。
然后，在新文件 `mmpose/core/optimizer/my_optimizer.py` 中实现该优化器：

```python
from .builder import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):

```

#### 2. 注册这个优化器

新优化器必须先导入主命名空间才能被成功调用。有两种实现方式。

- 修改 `mmpose/core/optimizer/__init__.py` 来导入

  新定义的优化器得在 `mmpose/core/optimizer/__init__.py` 中被导入，注册器才能发现并添加它。

```python
from .my_optimizer import MyOptimizer
```

- 在配置文件中使用 `custom_imports` 手动导入

```python
custom_imports = dict(imports=['mmpose.core.optimizers.my_optimizer'], allow_failed_imports=False)
```

在程序运行之初，库 `mmpose.core.optimizer.my_optimizer` 将会被导入。此时类 `MyOptimizer` 会自动注册。
注意只有包含类 `MyOptimizer` 的库才能被导入。 `mmpose.core.optimizer.my_optimizer.MyOptimizer` **不可以**被直接导入。

#### 3. 在配置文件中指定优化器

在新优化器 `MyOptimizer` 注册之后，它可以在配置文件中通过 `optimizer` 调用。
在配置文件中，优化器通过 `optimizer` 以如下方式指定：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

如果要使用自己实现的新优化器 `MyOptimizer`，可以进行如下修改：

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### 自定义优化器构造器

有些模型可能需要在优化器里对一些特别参数进行设置，例如批归一化层的权重衰减系数。
用户可以通过自定义优化器构造器来实现这些精细参数的调整。

```python
from mmcv.utils import build_from_cfg

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmpose.utils import get_root_logger
from .my_optimizer import MyOptimizer


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor:

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        pass

    def __call__(self, model):

        return my_optimizer
```

[这里](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/optimizer/default_constructor.py#L11)是默认优化器构造器的实现。它还可以用作新的优化器构造器的模板。

### 更多设置

有些优化器没有实现的功能可以通过优化器构造器（例如对不同权重设置不同学习率）或者钩子实现。
我们列出了一些用于稳定、加速训练的常用设置。欢迎通过PR、issue提出更多这样的设置。

- __使用梯度截断来稳定训练__：
  有些模型需要梯度截断来使梯度数值保持在某个范围，以让训练过程更加稳定。例如：

  ```python
  optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
  ```

- __使用动量策略加速模型收敛__
  我们支持根据学习率来修改模型动量的动量调度器。它可以让模型收敛更快。
  动量调度器通常和学习率调度器一起使用。例如3D检测中使用下面的配置来加速收敛。
  更多细节可以参考 [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327) 和 [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130) 的实现。

  ```python
  lr_config = dict(
      policy='cyclic',
      target_ratio=(10, 1e-4),
      cyclic_times=1,
      step_ratio_up=0.4,
  )
  momentum_config = dict(
      policy='cyclic',
      target_ratio=(0.85 / 0.95, 1),
      cyclic_times=1,
      step_ratio_up=0.4,
  )
  ```

## 自定义训练策略

我们默认使用的学习率变化策略为阶梯式衰减策略，即MMCV中的[`StepLRHook`](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L153)。
此外，我们还支持很多[学习率变化策略](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py)，例如余弦退火策略 `CosineAnnealing` 和多项式策略 `Poly`。其调用方式如下

- 多项式策略:

  ```python
  lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
  ```

- 余弦退火策略:

  ```python
  lr_config = dict(
      policy='CosineAnnealing',
      warmup='linear',
      warmup_iters=1000,
      warmup_ratio=1.0 / 10,
      min_lr_ratio=1e-5)
  ```

## 自定义工作流

我们推荐用户在每轮训练结束后对模型进行评估，即采用 `EpochEvalHook` 工作流。不过很多用户仍采用 `val` 工作流。

工作流是一个由（阶段，轮数）构成的列表，它规定了程序运行中不同阶段的顺序和轮数。默认的工作流为

```python
workflow = [('train', 1)]
```

即“训练 1 轮”。
有时候用户可能想要计算模型在验证集上的某些指标（例如损失、准确率）。此时可将工作流设定为

```python
[('train', 1), ('val', 1)]
```

即1轮训练后进行1轮验证，两者交替进行。

```{note}
1. 进行验证时，模型权重不会发生变化。
1. 配置文件中，参数 `total_epochs` 只控制训练轮数，不影响验证工作流
1. 工作流 `[('train', 1), ('val', 1)]` 和 `[('train', 1)]` 不会改变 `EpochEvalHook` 的行为。因为 `EpochEvalHook` 只在 `after_train_epoch` 中被调用。而验证工作流只会影响被 `after_val_epoch` 调用的钩子。
   因此，工作流 `[('train', 1), ('val', 1)]` 与 `[('train', 1)]` 唯一的差别就是运行程序会在每轮训练后计算模型在验证集上的损失。
```

## 自定义钩子

### 使用自己实现的钩子

#### 1. 定义一个新的钩子

下面的例子展示了如何定义一个新的钩子并将其用于训练。

```python
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
```

用户需要根据钩子的实际用途定义该钩子在 `before_run` 、`after_run` 、`before_epoch` 、`after_epoch` 、`before_iter` 以及 `after_iter` 中的行为。

#### 2. 注册这个新的钩子

定义好钩子 `MyHook` 之后，我们需要将其导入。假设 `MyHook` 在文件 `mmpose/core/utils/my_hook.py` 中定义，则有两种方式可以导入：

- 通过修改 `mmpose/core/utils/__init__.py` 进行导入。

  新定义的模块需要被导入到 `mmpose/core/utils/__init__.py` 才能被注册器找到并添加：

```python
from .my_hook import MyHook
```

- 在配置文件中使用 `custom_imports` 手动导入

```python
custom_imports = dict(imports=['mmpose.core.utils.my_hook'], allow_failed_imports=False)
```

#### 3. 修改配置文件

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

用户可以通过将钩子的参数 `priority` 设置为 `'NORMAL'` 或 `'HIGHEST'` 来设定它的优先级

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

钩子在注册时，其优先级默认为 `NORMAL`。

### 使用MMCV中的钩子

用户可以直接修改配置文件来调用MMCV中已实现的钩子

```python
mmcv_hooks = [
    dict(type='MMCVHook', a=a_value, b=b_value, priority='NORMAL')
]
```

### 修改默认的运行钩子

有部分常用钩子没有通过 `custom_hooks` 注册。在导入MMCV时，它们会自动注册。这些钩子包括：

- log_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

这些钩子中，只有日志钩子的优先级为 `VERY_LOW`，其他钩子的优先级都是 `NORMAL`。
前面的教程已经讲述了如何修改 `optimizer_config` 、`momentum_config` 、`lr_config`。这里我们介绍如何修改 `log_config` 、`checkpoint_config` 、`evaluation`。

#### 模型权重文件配置

MMCV的运行程序会使用 `checkpoint_config` 来初始化 [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/hooks/checkpoint.py#L9)。

```python
checkpoint_config = dict(interval=1)
```

用户可以通过设置 `max_keep_ckpts` 来保存有限的模型权重文件；通过设置 `save_optimizer` 以决定是否保存优化器的状态。
[这份文档](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)介绍了更多参数的细节。

#### 日志配置

日志配置 `log_config` 可以设置多个日志钩子，并且可以设定记录间隔。目前MMCV支持的日志钩子包括 `WandbLoggerHook` 、`MlflowLoggerHook` 、`TensorboardLoggerHook`。
[这份文档](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook)介绍了更多日志钩子的使用细节。

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### 测试配置

测试配置 `evaluation` 可以用来初始化 [`EvalHook`](https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/evaluation/eval_hooks.py#L11)。
除了参数 `interval`，其他参数（例如 `metric`）会被传递给 `dataset.evaluate()`。

```python
evaluation = dict(interval=1, metric='mAP')
```
