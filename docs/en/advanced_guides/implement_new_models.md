# Implement New Models

This tutorial will introduce how to implement your own models in MMPose. After summarizing, we split the need to implement new models into two categories:

1. Based on the algorithm paradigm supported by MMPose, customize the modules (backbone, neck, head, codec, etc.) in the model
2. Implement new algorithm paradigm

## Basic Concepts

What you want to implement is one of the above, and this section is important to you because it is the basic principle of building models in the OpenMMLab.

In MMPose, all the code related to the implementation of the model structure is stored in the [models directory](https://github.com/open-mmlab/mmpose/tree/main/mmpose/models) :

```shell
mmpose
|----models
     |----backbones             #
     |----data_preprocessors    # image normalization
     |----heads                 #
     |----losses                # loss functions
     |----necks                 #
     |----pose_estimators       # algorithm paradigm
     |----utils                 #
```

You can refer to the following flow chart to locate the module you need to implement:

![image](https://github.com/open-mmlab/mmpose/assets/13503330/f4eeb99c-e2a1-4907-9d46-f110c51f0814)

## Pose Estimatiors

In pose estimatiors, we will define the inference process of a model, and decode the model output results in `predict()`, first transform it from `output space` to `input image space` using the [codec](./codecs.md), and then combine the meta information to transform to `original image space`.

![pose_estimator_en](https://github.com/open-mmlab/mmpose/assets/13503330/0764baab-41c7-4a1d-ab64-5d7f9dfc8eec)

Currently, MMPose supports the following types of pose estimator:

1. [Top-down](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/pose_estimators/topdown.py): The input of the pose model is a cropped single target (animal, human body, human face, human hand, plant, clothes, etc.) image, and the output is the key point prediction result of the target
2. [Bottom-up](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/pose_estimators/bottomup.py): The input of the pose model is an image containing any number of targets, and the output is the key point prediction result of all targets in the image
3. [Pose Lifting](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/pose_estimators/pose_lifter.py): The input of the pose model is a 2D keypoint coordinate array, and the output is a 3D keypoint coordinate array

If the model you want to implement does not belong to the above algorithm paradigm, then you need to inherit the [BasePoseEstimator](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/pose_estimators/base.py) class to define your own algorithm paradigm.

## Backbones

If you want to implement a new backbone network, you need to create a new file in the [backbones directory](https://github.com/open-mmlab/mmpose/tree/main/mmpose/models/backbones) to define it.

The new backbone network needs to inherit the [BaseBackbone](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/base_backbone.py) class, and there is no difference in other aspects from inheriting `nn.Module` to create.

After completing the implementation of the backbone network, you need to use `MODELS` to register it:

```Python3
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YourNewBackbone(BaseBackbone):
```

Finally, please remember to import your new backbone network in [\_\_init\_\_.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/__init__.py) .

## Heads

The addition of a new prediction head is similar to the backbone network process. You need to create a new file in the [heads directory](https://github.com/open-mmlab/mmpose/tree/main/mmpose/models/heads) to define it, and then inherit [BaseHead](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/base_head.py) .

One thing to note is that in MMPose, the loss function is calculated in the Head. According to the different training and evaluation stages, `loss()` and `predict()` are executed respectively.

In `predict()`, the model will call the `decode()` method of the corresponding codec to transform the model output result from `output space` to `input image space`.

After completing the implementation of the prediction head, you need to use `MODELS` to register it:

```Python3
from mmpose.registry import MODELS
from ..base_head import BaseHead

@MODELS.register_module()
class YourNewHead(BaseHead):
```

Finally, please remember to import your new prediction head in [\_\_init\_\_.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/__init__.py).

### Head with Keypoints Visibility Prediction

Many models predict keypoint visibility based on confidence in coordinate predictions. However, this approach is suboptimal. Our [VisPredictHead](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/models/heads/hybrid_heads/vis_head.py) wrapper enables heads to directly predict keypoint visibility from ground truth training data, improving reliability. To add visibility prediction, wrap your head module with VisPredictHead in the config file.

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

To implement such a head module wrapper, we only need to inherit [BaseHead](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/base_head.py), then pass the pose head configuration in `__init__()` and instantiate it through `MODELS.build()`. As shown below:

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

Then you can implement other parts of the code as a normal head.
