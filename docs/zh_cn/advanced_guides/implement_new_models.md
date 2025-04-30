# 实现新模型

本教程将介绍如何在 MMPose 中实现你自己的模型。我们经过总结，将实现新模型这一需求拆分为两类：

1. 基于 MMPose 中已支持的算法范式，对模型中的模块（骨干网络、颈部、预测头、编解码器等）进行自定义
2. 实现新的算法范式

## 基础知识

不论你想实现的模型是以上哪一种，这一节的内容都对你很重要，因为它是 OpenMMLab 系列算法库构建模型的基本原则。
在 MMPose 中，所有与模型结构实现相关的代码都存放在 [models 目录](https://github.com/open-mmlab/mmpose/tree/main/mmpose/models)下：

```shell
mmpose
|----models
     |----backbones             # 骨干网络
     |----data_preprocessors    # 数据预处理，如：图片归一化
     |----heads                 # 预测头
     |----losses                # 损失函数
     |----necks                 # 颈部
     |----pose_estimators       # 姿态估计算法范式
     |----utils                 # 工具方法
```

你可以参考以下流程图来定位你所需要实现的模块：

![image](https://github.com/open-mmlab/mmpose/assets/13503330/f4eeb99c-e2a1-4907-9d46-f110c51f0814)

## 姿态估计算法范式

在姿态估计范式中，我们会定义一个模型的推理流程，并在 `predict()` 中对模型输出结果进行解码，先将其从 `输出尺度空间` 用 [编解码器](./codecs.md) 变换到 `输入图片空间`，然后再结合元信息变换到 `原始图片空间`。

![pose_estimator_cn](https://github.com/open-mmlab/mmpose/assets/13503330/0c048f66-b889-4268-937f-71b8753b505f)

当前 MMPose 已支持以下几类算法范式：

1. [Top-down](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/pose_estimators/topdown.py)：Pose 模型的输入为经过裁剪的单个目标（动物、人体、人脸、人手、植物、衣服等）图片，输出为这个目标的关键点预测结果
2. [Bottom-up](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/pose_estimators/bottomup.py)：Pose 模型的输入为包含任意个目标的图片，输出为图片中所有目标的关键点预测结果
3. [Pose Lifting](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/pose_estimators/pose_lifter.py)：Pose 模型的输入为 2D 关键点坐标数组，输出为 3D 关键点坐标数组

如果你要实现的模型不属于以上算法范式，那么你需要继承 [BasePoseEstimator](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/pose_estimators/base.py) 类来定义你自己的算法范式。

## 骨干网络

如果希望实现一个新的骨干网络，你需要在 [backbones 目录](https://github.com/open-mmlab/mmpose/tree/main/mmpose/models/backbones) 下新建一个文件进行定义。

新建的骨干网络需要继承 [BaseBackbone](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/base_backbone.py) 类，其他方面与你继承 nn.Module 来创建没有任何不同。

在完成骨干网络的实现后，你需要使用 `MODELS` 来对其进行注册：

```Python3
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YourNewBackbone(BaseBackbone):
```

最后，请记得在 [backbones/\_\_init\_\_.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/__init__.py) 中导入你的新骨干网络。

## 预测头部

新的预测头部的加入与骨干网络流程类似，你需要在 [heads 目录](https://github.com/open-mmlab/mmpose/tree/main/mmpose/models/heads) 下新建一个文件进行定义，然后继承 [BaseHead](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/base_head.py)。

需要特别注意的一点是，在 MMPose 中会在 Head 里进行损失函数的计算。根据训练与评测阶段的不同，分别执行 `loss()` 和 `predict()`。

在 `predict()` 中，模型会调用对应编解码器的 `decode()` 方法，将模型输出的结果从 `输出尺度空间` 转换到 `输入图片空间` 。

在完成预测头部的实现后，你需要使用 `MODELS` 来对其进行注册：

```Python3
from mmpose.registry import MODELS
from ..base_head import BaseHead

@MODELS.register_module()
class YourNewHead(BaseHead):
```

最后，请记得在 [heads/\_\_init\_\_.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/__init__.py) 中导入你的新预测头部。

### 关键点可见性预测头部

许多模型都是通过对关键点坐标预测的置信度来判断关键点的可见性的。然而，这种解决方案并非最优。我们提供了一个叫做 [VisPredictHead](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/models/heads/hybrid_heads/vis_head.py) 的头部模块包装器，使得头部模块能够直接预测关键点的可见性。这个包装器是用训练数据中关键点可见性真值来训练的。因此，其预测会更加可靠。用户可以通过修改配置文件来对自己的头部模块加上这个包装器。下面是一个例子:

```python
model=dict(
     ...
     head=dict(
          type='VisPredictHead',
          loss=dict(
               type='BCELoss',
               use_target_weight=True,
               use_sigmoid=True,
               loss_weight=1e-3),
          pose_cfg=dict(
               type='HeatmapHead',
               in_channels=2048,
               out_channels=17,
               loss=dict(type='KeypointMSELoss', use_target_weight=True),
               decoder=codec)),
     ...
)
```

要实现这样一个预测头部模块包装器，我们只需要像定义正常的预测头部一样，继承 [BaseHead](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/base_head.py)，然后在 `__init__()` 中传入关键点定位的头部配置，并通过 `MODELS.build()` 进行实例化。如下所示：

```python
@MODELS.register_module()
class VisPredictHead(BaseHead):
    """VisPredictHead must be used together with other heads. It can predict
    keypoints coordinates of and their visibility simultaneously. In the
    current version, it only supports top-down approaches.

    Args:
        pose_cfg (Config): Config to construct keypoints prediction head
        loss (Config): Config for visibility loss. Defaults to use
            :class:`BCELoss`
        use_sigmoid (bool): Whether to use sigmoid activation function
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(self,
                 pose_cfg: ConfigType,
                 loss: ConfigType = dict(
                     type='BCELoss', use_target_weight=False,
                     use_sigmoid=True),
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = pose_cfg['in_channels']
        if pose_cfg.get('num_joints', None) is not None:
            self.out_channels = pose_cfg['num_joints']
        elif pose_cfg.get('out_channels', None) is not None:
            self.out_channels = pose_cfg['out_channels']
        else:
            raise ValueError('VisPredictHead requires \'num_joints\' or'
                             ' \'out_channels\' in the pose_cfg.')

        self.loss_module = MODELS.build(loss)

        self.pose_head = MODELS.build(pose_cfg)
        self.pose_cfg = pose_cfg

        self.use_sigmoid = loss.get('use_sigmoid', False)

        modules = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_channels, self.out_channels)
        ]
        if self.use_sigmoid:
            modules.append(nn.Sigmoid())

        self.vis_head = nn.Sequential(*modules)
```

然后你只需要像一个普通的预测头一样继续实现其余部分即可。
