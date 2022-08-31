# Overview

This chapter will introduce you to the overall framework of MMPose and provide links to detailed tutorials.

## What is MMPose

MMPose is a Pytorch-based pose estimation open-source toolkit, a member of the [OpenMMLab Project](https://github.com/open-mmlab). It contains a rich set of algorithms for 2d multi-person human pose estimation, 2d hand pose estimation, 2d face landmark detection, 133 keypoint whole-body human pose estimation, fashion landmark detection and animal pose estimation as well as related components and modules, below is its overall framework.

MMPose consists of **7** main components:

- **apis** provides high-level APIs for model inference.
- **structures** provides data structures like bbox, keypoint and PoseDataSample.
- **datasets** supports various datasets for pose estimation and keypoint detection
  - **transforms** contains a lot of useful data augmentation transforms.
- **codecs** provides the encoder and decoder for target generation and output decoding.
- **models** is the most vital part for pose estimators and contains different components of a estimator.
  - **pose_estimators** defines all of the estimation model classes.
  - **data_preprocessors** is for preprocessing the input data  of the model
  - **backbones** contains a bunch of backbone networks.
  - **necks** contains various neck components.
  - **heads** contains various prediction heads that perform pose estimation.
  - **losses** contains various loss functions
- **engine** is a part for runtime components.
  - **hooks** provides various hooks of the runner
- **evaluation** provides different metrics for evaluating model performance.
- **visualization** is for visualizing skeleton and heatmaps

## How to Use this Guide

We have prepared detailed guidelines for all types of users:

1. For installation instrunctions:
   - [Installation](./installation.md)
2. For the basic usage of MMPose:
   - [Quick Run](./quick_run.md)
   - [Inference](./user_guides/inference.md)
3. For users who want to learn more about components of MMPose:
   - [Configs](./user_guides/configs.md)
   - [Prepare Datasets](./user_guides/prepare_datasets.md)
   - [Codecs](./user_guides/codecs.md)
   - [Train & Test](./user_guides/train_and_test.md)
   - [Visualization](./user_guides/visualization.md)
   - [Useful Tools](./user_guides/useful_tools.md)
4. For developers who wish to develop based on MMPose:
   - [Migration Guide](./migration.md)
5. For researchers and developers who are willing to contribute to MMPose:
   - [Contribution Guide](./notes/contribution_guide.md)
6. For some common issues, we provide a FAQ list:
   - [FAQ](./notes/faq.md)
