<div align="center">
    <img src="resources/mmpose-logo.png" width="400"/>
</div>

## Introduction

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://badge.fury.io/py/mmpose.svg)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.3+**.

<div align="center">
    <img src="demo/demo_coco.gif" width="600px" alt><br>
    COCO 17-keypoint pose estimation
</div>
<div align="center">
<img src="https://user-images.githubusercontent.com/9464825/95552839-00a61080-0a40-11eb-818c-b8dad7307217.gif" width="600px" alt><br>

133-keypoint whole-body pose estimation ([full HD version](https://www.youtube.com/watch?v=pIJpQg8mXUU))

</div>

### Major Features

- **Support diverse tasks**

  We support a wide spectrum of mainstream human pose analysis tasks in current research community, including 2d multi-person human pose estimation, 2d hand pose estimation, 133 keypoint whole-body human pose estimation, fashion landmark detection and 3d human mesh recovery.

- **Higher efficiency and higher accuracy**

  MMPose implements multiple state-of-the-art (SOTA) deep learning models, including both top-down & bottom-up approaches. We achieve faster training speed and higher accuracy than other popular codebases, such as [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
  See [benchmark.md](docs/benchmark.md) for more information.

- **Support for various datasets**

  The toolbox directly supports multiple popular and representative datasets, COCO, AIC, MPII, MPII-TRB, OCHuman etc.
  See [data_preparation.md](docs/data_preparation.md) for more information.

- **Well designed, tested and documented**

  We decompose MMPose into different components and one can easily construct a customized
  pose estimation framework by combining different modules.
  We provide detailed documentation and API reference, as well as unittests.

## [Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html)

Supported backbones for human pose estimation:

- [x] [AlexNet](configs/top_down/alexnet/README.md)
- [x] [VGG](configs/top_down/vgg/README.md)
- [x] [HRNet](configs/top_down/hrnet/README.md)
- [x] [MobilenetV2](configs/top_down/mobilenet_v2/README.md)
- [x] [ResNet](configs/top_down/resnet/README.md)
- [x] [ResNetV1D](configs/top_down/resnetv1d/README.md)
- [x] [ResNeSt](configs/top_down/resnest/README.md)
- [x] [ResNext](configs/top_down/resnext/README.md)
- [x] [SCNet](configs/top_down/scnet/README.md)
- [x] [SEResNet](configs/top_down/seresnet/README.md)
- [x] [ShufflenetV1](configs/top_down/shufflenet_v1/README.md)
- [x] [ShufflenetV2](configs/top_down/shufflenet_v2/README.md)

Supported methods for human pose estimation:

- [x] [CPM](configs/top_down/cpm/README.md)
- [x] [SimpleBaseline](configs/top_down/resnet/README.md)
- [x] [HRNet](configs/top_down/hrnet/README.md)
- [x] [Hourglass](configs/top_down/hourglass/README.md)
- [x] [SCNet](configs/top_down/scnet/README.md)
- [x] [Associative Embedding](configs/bottom_up/hrnet/README.md)
- [x] [HigherHRNet](configs/bottom_up/higherhrnet/README.md)
- [x] [DarkPose](configs/top_down/darkpose/README.md)
- [x] [UDP](configs/top_down/udp/README.md)
- [x] [MSPN](configs/top_down/mspn/README.md)
- [x] [RSN](configs/top_down/rsn/README.md)

Supported [datasets](https://mmpose.readthedocs.io/en/latest/datasets.html):

- [x] [COCO](http://cocodataset.org/)
- [x] [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/)
- [x] [MPII](http://human-pose.mpi-inf.mpg.de/)
- [x] [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)
- [x] [AI Challenger](https://github.com/AIChallenger/AI_Challenger_2017)
- [x] [OCHuman](https://github.com/liruilong940607/OCHumanApi)
- [x] [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)
- [x] [PoseTrack18](https://posetrack.net/users/download.php)
- [x] [MHP](https://lv-mhp.github.io/dataset)
- [x] [sub-JHMDB](http://jhmdb.is.tue.mpg.de/dataset)
- [x] [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)
- [x] [FreiHand](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
- [x] [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html)
- [x] [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/)
- [x] [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
- [x] [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [**model zoo**](https://mmpose.readthedocs.io/en/latest/modelzoo.html) page.
We will keep up with the latest progress of the community, and support more popular algorithms and frameworks.

If you have any feature requests, please feel free to leave a comment in [Issues](https://github.com/open-mmlab/mmpose/issues/9).

## Benchmark

We demonstrate the superiority of our MMPose framework in terms of speed and accuracy on the standard COCO keypoint detection benchmark.

| Model | Input size| MMPose (s/iter) | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) (s/iter) | MMPose (mAP) | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) (mAP) |
| :--- | :---------------: | :---------------: |:--------------------: | :----------------------------: | :-----------------: |
| resnet_50  | 256x192  | **0.28** | 0.64 | **0.718** | 0.704 |
| resnet_50  | 384x288  | **0.81** | 1.24 | **0.731** | 0.722 |
| resnet_101 | 256x192  | **0.36** | 0.84 | **0.726** | 0.714 |
| resnet_101 | 384x288  | **0.79** | 1.53 | **0.748** | 0.736 |
| resnet_152 | 256x192  | **0.49** | 1.00 | **0.735** | 0.720 |
| resnet_152 | 384x288  | **0.96** | 1.65 | **0.750** | 0.743 |
| hrnet_w32  | 256x192  | **0.54** | 1.31 | **0.746** | 0.744 |
| hrnet_w32  | 384x288  | **0.76** | 2.00 | **0.760** | 0.758 |
| hrnet_w48  | 256x192  | **0.66** | 1.55 | **0.756** | 0.751 |
| hrnet_w48  | 384x288  | **1.23** | 2.20 | **0.767** | 0.763 |

More details about the benchmark are available on [benchmark.md](docs/benchmark.md).

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMPose.
There are also tutorials: [finetune model](tutorials/1_finetune.md),
[add new dataset](tutorials/2_new_dataset.md), [customize data pipelines](tutorials/3_data_pipeline.md),
[add new modules](tutorials/4_new_modules.md), [export a model to ONNX](tutorials/5_export_model.md) and [customize runtime settings](tutorials/6_customize_runtime.md).

## FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMPose. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmpose/blob/master/.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMPose is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
