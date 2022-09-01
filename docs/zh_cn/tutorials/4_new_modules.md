# 教程 4: 增加新的模块

## 自定义优化器

在本教程中，我们将介绍如何为项目定制优化器.
假设想要添加一个名为 `MyOptimizer` 的优化器，它有 `a`，`b` 和 `c` 三个参数。
那么首先需要在一个文件中实现该优化器，例如 `mmpose/core/optimizer/my_optimizer.py`：

```python
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

然后需要将其添加到 `mmpose/core/optimizer/__init__.py` 中，从而让注册器可以找到这个新的优化器并添加它：

```python
from .my_optimizer import MyOptimizer
```

之后，可以在配置文件的 `optimizer` 字段中使用 `MyOptimizer`。
在配置中，优化器由 `optimizer` 字段所定义，如下所示：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

若要使用自己新定义的优化器，可以将字段修改为：

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

我们已经支持使用 PyTorch 实现的所有优化器，
只需要更改配置文件的 `optimizer` 字段。
例如：若用户想要使用`ADAM`优化器，只需要做出如下修改，虽然这会造成网络效果下降。

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

用户可以直接根据 [PyTorch API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim)
对参数进行设置。

## 自定义优化器构造器

某些模型可能对不同层的参数有特定的优化设置，例如 BatchNorm 层的权值衰减。
用户可以通过自定义优化器构造函数来进行这些细粒度的参数调整。

```python
from mmcv.utils import build_from_cfg

from mmcv.runner import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmpose.utils import get_root_logger
from .cocktail_optimizer import CocktailOptimizer


@OPTIMIZER_BUILDERS.register_module()
class CocktailOptimizerConstructor:

    def __init__(self, optimizer_cfg, paramwise_cfg=None):

    def __call__(self, model):

        return my_optimizer

```

## 开发新组件

MMPose 将模型组件分为 3 种基础模型：

- 检测器（detector）：整个检测器模型流水线，通常包含一个主干网络（backbone）和关键点头（keypoint_head）。
- 主干网络（backbone）：通常为一个用于提取特征的 FCN 网络，例如 ResNet，HRNet。
- 关键点头（keypoint_head）：用于姿势估计的组件，通常包括一系列反卷积层。

1. 创建一个新文件 `mmpose/models/backbones/my_model.py`.

```python
import torch.nn as nn

from ..builder import BACKBONES

@BACKBONES.register_module()
class MyModel(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass
```

2. 在 `mmpose/models/backbones/__init__.py` 中导入新的主干网络.

```python
from .my_model import MyModel
```

3. 创建一个新文件 `mmpose/models/keypoint_heads/my_head.py`.

用户可以通过继承 `nn.Module` 编写一个新的关键点头，
并重写 `init_weights(self)` 和 `forward(self, x)` 方法。

```python
from ..builder import HEADS


@HEADS.register_module()
class MyHead(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):
        pass

    def init_weights(self):
        pass
```

4. 在 `mmpose/models/keypoint_heads/__init__.py` 中导入新的关键点头

```python
from .my_head import MyHead
```

5. 在配置文件中使用它。

对于自顶向下的 2D 姿态估计模型，我们将模型类型设置为 `TopDown`。

```python
model = dict(
    type='TopDown',
    backbone=dict(
        type='MyModel',
        arg1=xxx,
        arg2=xxx),
    keypoint_head=dict(
        type='MyHead',
        arg1=xxx,
        arg2=xxx))
```

### 添加新的损失函数

假设用户想要为关键点估计添加一个名为 `MyLoss`的新损失函数。
为了添加一个新的损失函数，用户需要在 `mmpose/models/losses/my_loss.py` 下实现该函数。
其中，装饰器 `weighted_loss` 使损失函数能够为每个元素加权。

```python
import torch
import torch.nn as nn

from mmpose.models import LOSSES

def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    loss = torch.mean(loss)
    return loss

@LOSSES.register_module()
class MyLoss(nn.Module):

    def __init__(self, use_target_weight=False):
        super(MyLoss, self).__init__()
        self.criterion = my_loss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred * target_weight[:, idx],
                    heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
```

之后，用户需要把它添加进 `mmpose/models/losses/__init__.py`。

```python
from .my_loss import MyLoss, my_loss

```

若要使用新的损失函数，可以修改模型中的 `loss_keypoint` 字段。

```python
loss_keypoint=dict(type='MyLoss', use_target_weight=False)
```
