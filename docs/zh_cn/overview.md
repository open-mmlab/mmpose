# 概述

本章将向你介绍 MMPose 的整体框架，并提供详细的教程链接。

## 什么是 MMPose

![overview](https://user-images.githubusercontent.com/13503330/191004511-508d3ec6-9ead-4c52-a522-4d9aa1f26027.png)

MMPose 是一款基于 Pytorch 的姿态估计开源工具箱，是 OpenMMLab 项目的成员之一，包含了丰富的 2D 多人姿态估计、2D 手部姿态估计、2D 人脸关键点检测、133关键点全身人体姿态估计、动物关键点检测、服饰关键点检测等算法以及相关的组件和模块，下面是它的整体框架：

MMPose 由 **8** 个主要部分组成，apis、structures、datasets、codecs、models、engine、evaluation 和 visualization。

- **apis** 提供用于模型推理的高级 API

- **structures** 提供 bbox、keypoint 和 PoseDataSample 等数据结构

- **datasets** 支持用于姿态估计的各种数据集

  - **transforms** 包含各种数据增强变换

- **codecs** 提供姿态编解码器：编码器用于将姿态信息（通常为关键点坐标）编码为模型学习目标（如热力图），解码器则用于将模型输出解码为姿态估计结果

- **models** 以模块化结构提供了姿态估计模型的各类组件

  - **pose_estimators** 定义了所有姿态估计模型类
  - **data_preprocessors** 用于预处理模型的输入数据
  - **backbones** 包含各种骨干网络
  - **necks** 包含各种模型颈部组件
  - **heads** 包含各种模型头部
  - **losses** 包含各种损失函数

- **engine** 包含与姿态估计任务相关的运行时组件

  - **hooks** 提供运行时的各种钩子

- **evaluation** 提供各种评估模型性能的指标

- **visualization** 用于可视化关键点骨架和热力图等信息

## 如何使用本指南

针对不同类型的用户，我们准备了详细的指南：

1. 安装说明：

   - [安装](./installation.md)

2. MMPose 的基本使用方法：

   - [20 分钟上手教程](./guide_to_framework.md)
   - [Demos](./demos.md)
   - [模型推理](./user_guides/inference.md)
   - [配置文件](./user_guides/configs.md)
   - [准备数据集](./user_guides/prepare_datasets.md)
   - [训练与测试](./user_guides/train_and_test.md)

3. 对于希望基于 MMPose 进行开发的研究者和开发者：

   - [编解码器](./advanced_guides/codecs.md)
   - [数据流](./advanced_guides/dataflow.md)
   - [实现新模型](./advanced_guides/implement_new_models.md)
   - [自定义数据集](./advanced_guides/customize_datasets.md)
   - [自定义数据变换](./advanced_guides/customize_transforms.md)
   - [自定义优化器](./advanced_guides/customize_optimizer.md)
   - [自定义日志](./advanced_guides/customize_logging.md)
   - [模型部署](./advanced_guides/how_to_deploy.md)
   - [模型分析工具](./advanced_guides/model_analysis.md)
   - [迁移指南](./migration.md)

4. 对于希望加入开源社区，向 MMPose 贡献代码的研究者和开发者：

   - [参与贡献代码](./contribution_guide.md)

5. 对于使用过程中的常见问题：

   - [FAQ](./faq.md)
