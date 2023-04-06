# Overview

This chapter will introduce you to the overall framework of MMPose and provide links to detailed tutorials.

## What is MMPose

![overview](https://user-images.githubusercontent.com/13503330/191004511-508d3ec6-9ead-4c52-a522-4d9aa1f26027.png)

MMPose is a Pytorch-based pose estimation open-source toolkit, a member of the [OpenMMLab Project](https://github.com/open-mmlab). It contains a rich set of algorithms for 2d multi-person human pose estimation, 2d hand pose estimation, 2d face landmark detection, 133 keypoint whole-body human pose estimation, fashion landmark detection and animal pose estimation as well as related components and modules, below is its overall framework.

MMPose consists of **8** main components:

- **apis** provides high-level APIs for model inference
- **structures** provides data structures like bbox, keypoint and PoseDataSample
- **datasets** supports various datasets for pose estimation
  - **transforms** contains a lot of useful data augmentation transforms
- **codecs** provides pose encoders and decoders: an encoder encodes poses (mostly keypoints) into learning targets (e.g. heatmaps), and a decoder decodes model outputs into pose predictions
- **models** provides all components of pose estimation models in a modular structure
  - **pose_estimators** defines all pose estimation model classes
  - **data_preprocessors** is for preprocessing the input data of the model
  - **backbones** provides a collection of backbone networks
  - **necks** contains various neck modules
  - **heads** contains various prediction heads that perform pose estimation
  - **losses** contains various loss functions
- **engine** provides runtime components related to pose estimation
  - **hooks** provides various hooks of the runner
- **evaluation** provides metrics for evaluating model performance
- **visualization** is for visualizing skeletons, heatmaps and other information

## How to Use this Guide

We have prepared detailed guidelines for all types of users:

1. For installation instrunctions:

   - [Installation](./installation.md)

2. For the basic usage of MMPose:

   - [A 20-minute Tour to MMPose](./guide_to_framework.md)
   - [Demos](./demos.md)
   - [Inference](./user_guides/inference.md)
   - [Configs](./user_guides/configs.md)
   - [Prepare Datasets](./user_guides/prepare_datasets.md)
   - [Train and Test](./user_guides/train_and_test.md)

3. For developers who wish to develop based on MMPose:

   - [Learn about Codecs](./advanced_guides/codecs.md)
   - [Dataflow in MMPose](./advanced_guides/dataflow.md)
   - [Implement New Models](./advanced_guides/implement_new_models.md)
   - [Customize Datasets](./advanced_guides/customize_datasets.md)
   - [Customize Data Transforms](./advanced_guides/customize_transforms.md)
   - [Customize Optimizer](./advanced_guides/customize_optimizer.md)
   - [Customize Logging](./advanced_guides/customize_logging.md)
   - [How to Deploy](./advanced_guides/how_to_deploy.md)
   - [Model Analysis](./advanced_guides/model_analysis.md)
   - [Migration Guide](./migration.md)

4. For researchers and developers who are willing to contribute to MMPose:

   - [Contribution Guide](./contribution_guide.md)

5. For some common issues, we provide a FAQ list:

   - [FAQ](./faq.md)
