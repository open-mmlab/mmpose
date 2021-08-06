# Tutorial 4: Adding New Modules

## Customize optimizer

A customized optimizer could be defined as following.
Assume you want to add a optimizer named as `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to first implement the new optimizer in a file, e.g., in `mmpose/core/optimizer/my_optimizer.py`:

```python
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)

```

Then add this module in `mmpose/core/optimizer/__init__.py` thus the registry will
find the new module and add it:

```python
from .my_optimizer import MyOptimizer
```

Then you can use `MyOptimizer` in `optimizer` field of config files.
In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

To use your own optimizer, the field can be changed as

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

We already support to use all the optimizers implemented by PyTorch, and the only modification is to change the `optimizer` field of config files.
For example, if you want to use `ADAM`, though the performance will drop a lot, the modification could be as the following.

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

The users can directly set arguments following the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

## Customize optimizer constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.
The users can do those fine-grained parameter tuning through customizing optimizer constructor.

```
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

### Develop new components

We basically categorize model components into 3 types.

- detectors: the whole pose detector model pipeline, usually contains a backbone and keypoint_head.
- backbone: usually an FCN network to extract feature maps, e.g., ResNet, HRNet.
- keypoint_head: the component for pose estimation task, usually contains some deconv layers.

1. Create a new file `mmpose/models/backbones/my_model.py`.

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

2. Import the module in `mmpose/models/backbones/__init__.py`.

```python
from .my_model import MyModel
```

3. Create a new file `mmpose/models/keypoint_heads/my_head.py`.

You can write a new keypoint head inherit from `nn.Module`,
and overwrite `init_weights(self)` and `forward(self, x)` method.

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

4. Import the module in `mmpose/models/keypoint_heads/__init__.py`

```python
from .my_head import MyHead
```

5. Use it in your config file.

For the top-down 2D pose estimation model, we set the module type as `TopDown`.

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

### Add new loss

Assume you want to add a new loss as `MyLoss`, for keypoints estimation.
To add a new loss function, the users need implement it in `mmpose/models/losses/my_loss.py`.
The decorator `weighted_loss` enable the loss to be weighted for each element.

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

Then the users need to add it in the `mmpose/models/losses/__init__.py`.

```python
from .my_loss import MyLoss, my_loss

```

To use it, modify the `loss_keypoint` field in the model.

```python
loss_keypoint=dict(type='MyLoss', use_target_weight=False)
```
