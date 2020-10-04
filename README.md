<div align="center">
    <img src="resources/mmpose-logo.png" width="400"/>
</div>

## Introduction

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.3+**.

<div align="center">
  <img src="demo/demo.gif" width="600px"/>
</div>


### Major Features

- **Support top-down & bottom-up approaches**

  MMPose implements multiple state-of-the-art (SOTA) deep learning models for human pose estimation, including both top-down and bottom-up approaches.

- **Higher efficiency and Higher Accuracy**

  We achieve faster training speed and higher accuracy than other popular codebases, such as [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
  See [benchmark.md](docs/benchmark.md) for more information.

- **Support for various datasets**

  The toolbox directly supports multiple datasets, COCO, AIC, MPII, MPII-TRB, OCHuman etc.
  See [data_preparation.md](docs/data_preparation.md) for more information.

- **Well designed, tested and documented**

  We decompose MMPose into different components and one can easily construct a customized
  pose estimation framework by combining different modules.
  We provide detailed documentation and API reference, as well as unittests.


## [Model Zoo](https://mmpose.readthedocs.io/en/latest/top_down_models.html)

Supported backbones for human pose estimation:

- [x] [AlexNet](configs/top_down/alexnet/README.md)
- [x] [HRNet](configs/top_down/hrnet/README.md)
- [x] [MobilenetV2](configs/top_down/mobilenet_v2/README.md)
- [x] [ResNet](configs/top_down/resnet/README.md)
- [x] [ResNetV1D](configs/top_down/resnetv1d/README.md)
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
- [x] [HigherHRNet](configs/bottom_up/higherhrnet/README.md)
- [x] [DarkPose](configs/top_down/darkpose/README.md)

Supported datasets:

- [x] [COCO](http://cocodataset.org/)
- [x] [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/)
- [x] [MPII](http://human-pose.mpi-inf.mpg.de/)
- [x] [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)
- [x] [AI Challenger](https://github.com/AIChallenger/AI_Challenger_2017)
- [x] [OCHuman](https://github.com/liruilong940607/OCHumanApi)
- [x] [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)
- [x] [H36m](http://vision.imar.ro/human3.6m/description.php)
- [x] [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)
- [x] [FreiHand](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
- [x] [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html)

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [**model zoo**](https://mmpose.readthedocs.io/en/latest/top_down_models.html) page.
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
There are also tutorials for [finetuning model](tutorials/finetune.md),
[adding new dataset](tutorials/new_dataset.md), [adding new modules](tutorials/new_modules.md).

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contributing

We appreciate all contributions to improve MMPose. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmpose/blob/master/.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMPose is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.
