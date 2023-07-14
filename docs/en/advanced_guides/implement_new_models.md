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

Finally, please remember to import your new backbone network in `[__init__.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/__init__.py)` .

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

Finally, please remember to import your new prediction head in `[__init__.py](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/__init__.py)` .
