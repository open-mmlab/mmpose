# 概述

本章将向你介绍 MMPose 的整体框架，并提供详细的教程链接。

## 什么是 MMPose

MMPose 是一款基于 Pytorch 的姿态估计开源工具箱，是 OpenMMLab 项目的成员之一，包含了丰富的2D多人姿态估计、2D手部姿态估计、2D人脸关键点检测、133关键点全身人体姿态估计、动物关键点检测、服饰关键点检测等算法以及相关的组件和模块，下面是它的整体框架：

MMPose 由 7 个主要部分组成，apis、structures、datasets、codecs、models、engine、评估和可视化。

- **apis** 为模型推理提供高级 API

- **structures** 提供bbox、keypoint和 PoseDataSample 等数据结构

- **datasets** 支持用于关键点检测和姿态估计的各种数据集

  - **transforms** 包含各种数据增强变换

- **codecs** 提供训练目标生成与模型输出解码所需的编码器和解码器

- **models** 是检测器最重要的部分，包含检测器的不同组件

  - **pose_estimators** 定义了所有姿态估计模型类
  - **data_preprocessors** 用于预处理模型的输入数据
  - **backbones** 包含各种骨干网络
  - **necks** 包含各种模型颈部组件
  - **heads** 包含各种模型头部
  - **losses** 包含各种损失函数

- **engine** 是运行时组件的一部分

  - **hooks** 提供运行时的各种钩子

- **evaluation** 为评估模型性能提供不同的指标

- **visualization** 用于可视化关键点骨架和热度图

## 如何使用本指南

针对不同类型的用户，我们准备了详细的指南：

1. 安装说明：

   - [安装](./installation.md)

2. MMPose 的基本使用方法：

   - [快速上手](./quick_run.md)
   - [模型推理](./user_guides/inference.md)

3. 对于希望了解 MMPose 各个组件的用户：

   - [配置文件](./user_guides/configs.md)
   - [准备数据集](./user_guides/prepare_datasets.md)
   - [编解码器](./user_guides/codecs.md)
   - [训练与测试](./user_guides/train_and_test.md)
   - [可视化](./user_guides/visualization.md)
   - [常用工具](./user_guides/useful_tools.md)

4. 对于希望将自己的项目迁移到 MMPose 的开发者：

   - [迁移指南](./migration.md)

5. 对于希望加入开源社区，向 MMPose 贡献代码的研究者和开发者：

   - [参与贡献代码](./notes/contribution_guide.md)

6. 对于使用过程中的常见问题：

   - [FAQ](./notes/faq.md)
