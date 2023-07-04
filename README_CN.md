<div align="center">
  <img src="resources/mmpose-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/latest/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://img.shields.io/pypi/v/mmpose)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/main/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

[📘文档](https://mmpose.readthedocs.io/zh_CN/latest/) |
[🛠️安装](https://mmpose.readthedocs.io/zh_CN/latest/installation.html) |
[👀模型库](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo.html) |
[📜论文库](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html) |
[🆕更新日志](https://mmpose.readthedocs.io/zh_CN/latest/notes/changelog.html) |
[🤔报告问题](https://github.com/open-mmlab/mmpose/issues/new/choose) |
[🔥RTMPose](/projects/rtmpose/)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1072798105428299817" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## Introduction

[English](./README.md) | 简体中文

MMPose 是一款基于 PyTorch 的姿态分析的开源工具箱，是 [OpenMMLab](https://github.com/open-mmlab) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.8 以上**的版本。

https://user-images.githubusercontent.com/15977946/124654387-0fd3c500-ded1-11eb-84f6-24eeddbf4d91.mp4

<details close>
<summary><b>主要特性</b></summary>

- **支持多种人体姿态分析相关任务**

  MMPose 支持当前学界广泛关注的主流姿态分析任务：主要包括 2D多人姿态估计、2D手部姿态估计、2D人脸关键点检测、133关键点的全身人体姿态估计、3D人体形状恢复、服饰关键点检测、动物关键点检测等。
  具体请参考 [功能演示](demo/docs/zh_cn/)。

- **更高的精度和更快的速度**

  MMPose 复现了多种学界最先进的人体姿态分析模型，包括“自顶向下”和“自底向上”两大类算法。MMPose 相比于其他主流的代码库，具有更高的模型精度和训练速度。
  具体请参考 [基准测试](docs/en/notes/benchmark.md)（英文）。

- **支持多样的数据集**

  MMPose 支持了很多主流数据集的准备和构建，如 COCO、 MPII 等。 具体请参考 [数据集](docs/zh_cn/dataset_zoo)。

- **模块化设计**

  MMPose 将统一的人体姿态分析框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的人体姿态分析模型。

- **详尽的单元测试和文档**

  MMPose 提供了详尽的说明文档，API 接口说明，全面的单元测试，以供社区参考。

</details>

## 最新进展

- 我们支持了三个新的数据集:
  - (CVPR 2023) [Human-Art](https://github.com/IDEA-Research/HumanArt)
  - (CVPR 2022) [Animal Kingdom](https://github.com/sutdcv/Animal-Kingdom)
  - (AAAI 2020) [LaPa](https://github.com/JDAI-CV/lapa-dataset/)

![yolox-pose_intro](https://user-images.githubusercontent.com/26127467/226655503-3cee746e-6e42-40be-82ae-6e7cae2a4c7e.jpg)

- 欢迎使用 [*MMPose 项目*](/projects/README.md)。在这里，您可以发现 MMPose 中的最新功能和算法，并且可以通过最快的方式与社区分享自己的创意和代码实现。向 MMPose 中添加新功能从此变得简单丝滑：

  - 提供了一种简单迅捷的方式，将新的算法、功能和应用添加到 MMPose 中
  - 更灵活的代码结构和风格，更少的限制，更简短的代码审核流程
  - 通过独立项目的形式，利用 MMPose 的强大功能，同时不被代码框架所束缚
  - 最新添加的项目包括：
    - [RTMPose](/projects/rtmpose/)
    - [YOLOX-Pose](/projects/yolox_pose/)
    - [MMPose4AIGC](/projects/mmpose4aigc/)
    - [Simple Keypoints](/projects/skps/)
  - 从简单的 [示例项目](/projects/example_project/) 开启您的 MMPose 代码贡献者之旅吧，让我们共同打造更好用的 MMPose！

<br/>

- 2023-07-04：MMPose [v1.1.0](https://github.com/open-mmlab/mmpose/releases/tag/v1.1.0) 正式发布了，主要更新包括:

  - 支持新数据集：Human-Art、Animal Kingdom、LaPa。
  - 支持新的配置文件风格，支持 IDE 跳转和搜索。
  - 提供更强性能的 RTMPose 模型。
  - 迁移 3D 姿态估计算法。
  - 加速推理脚本，全部 demo 脚本支持摄像头推理。

  请查看完整的 [版本说明](https://github.com/open-mmlab/mmpose/releases/tag/v1.1.0) 以了解更多 MMPose v1.1.0 带来的更新!

## 0.x / 1.x 迁移

MMPose v1.0.0 是一个重大更新，包括了大量的 API 和配置文件的变化。目前 v1.0.0 中已经完成了一部分算法的迁移工作，剩余的算法将在后续的版本中陆续完成，我们将在下面的列表中展示迁移进度。

<details close>
<summary><b>迁移进度</b></summary>

| 算法名称                          |  迁移进度   |
| :-------------------------------- | :---------: |
| MTUT (CVPR 2019)                  |             |
| MSPN (ArXiv 2019)                 |    done     |
| InterNet (ECCV 2020)              |             |
| DEKR (CVPR 2021)                  |    done     |
| HigherHRNet (CVPR 2020)           |             |
| DeepPose (CVPR 2014)              |    done     |
| RLE (ICCV 2021)                   |    done     |
| SoftWingloss (TIP 2021)           |    done     |
| VideoPose3D (CVPR 2019)           |    done     |
| Hourglass (ECCV 2016)             |    done     |
| LiteHRNet (CVPR 2021)             |    done     |
| AdaptiveWingloss (ICCV 2019)      |    done     |
| SimpleBaseline2D (ECCV 2018)      |    done     |
| PoseWarper (NeurIPS 2019)         |             |
| SimpleBaseline3D (ICCV 2017)      |    done     |
| HMR (CVPR 2018)                   |             |
| UDP (CVPR 2020)                   |    done     |
| VIPNAS (CVPR 2021)                |    done     |
| Wingloss (CVPR 2018)              |    done     |
| DarkPose (CVPR 2020)              |    done     |
| Associative Embedding (NIPS 2017) | in progress |
| VoxelPose (ECCV 2020)             |             |
| RSN (ECCV 2020)                   |    done     |
| CID (CVPR 2022)                   |    done     |
| CPM (CVPR 2016)                   |    done     |
| HRNet (CVPR 2019)                 |    done     |
| HRNetv2 (TPAMI 2019)              |    done     |
| SCNet (CVPR 2020)                 |    done     |

</details>

如果您使用的算法还没有完成迁移，您也可以继续使用访问 [0.x 分支](https://github.com/open-mmlab/mmpose/tree/0.x) 和 [旧版文档](https://mmpose.readthedocs.io/zh_CN/0.x/)

## 安装

关于安装的详细说明请参考[安装文档](https://mmpose.readthedocs.io/zh_CN/latest/installation.html)。

## 教程

我们提供了一系列简明的教程，帮助 MMPose 的新用户轻松上手使用：

1. MMPose 的基本使用方法：

   - [20 分钟上手教程](https://mmpose.readthedocs.io/zh_CN/latest/guide_to_framework.html)
   - [Demos](https://mmpose.readthedocs.io/zh_CN/latest/demos.html)
   - [模型推理](https://mmpose.readthedocs.io/zh_CN/latest/user_guides/inference.html)
   - [配置文件](https://mmpose.readthedocs.io/zh_CN/latest/user_guides/configs.html)
   - [准备数据集](https://mmpose.readthedocs.io/zh_CN/latest/user_guides/prepare_datasets.html)
   - [训练与测试](https://mmpose.readthedocs.io/zh_CN/latest/user_guides/train_and_test.html)

2. 对于希望基于 MMPose 进行开发的研究者和开发者：

   - [编解码器](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/codecs.html)
   - [数据流](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/dataflow.html)
   - [实现新模型](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/implement_new_models.html)
   - [自定义数据集](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/customize_datasets.html)
   - [自定义数据变换](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/customize_transforms.html)
   - [自定义优化器](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/customize_optimizer.html)
   - [自定义日志](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/customize_logging.html)
   - [模型部署](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/how_to_deploy.html)
   - [模型分析工具](https://mmpose.readthedocs.io/zh_CN/latest/advanced_guides/model_analysis.html)
   - [迁移指南](https://mmpose.readthedocs.io/zh_CN/latest/migration.html)

3. 对于希望加入开源社区，向 MMPose 贡献代码的研究者和开发者：

   - [参与贡献代码](https://mmpose.readthedocs.io/zh_CN/latest/contribution_guide.html)

4. 对于使用过程中的常见问题：

   - [FAQ](https://mmpose.readthedocs.io/zh_CN/latest/faq.html)

## 模型库

各个模型的结果和设置都可以在对应的 config（配置）目录下的 **README.md** 中查看。
整体的概况也可也在 [模型库](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo.html) 页面中查看。

<details close>
<summary><b>支持的算法</b></summary>

- [x] [DeepPose](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#deeppose-cvpr-2014) (CVPR'2014)
- [x] [CPM](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#cpm-cvpr-2016) (CVPR'2016)
- [x] [Hourglass](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#hourglass-eccv-2016) (ECCV'2016)
- [x] [SimpleBaseline3D](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#simplebaseline3d-iccv-2017) (ICCV'2017)
- [ ] [Associative Embedding](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#associative-embedding-nips-2017) (NeurIPS'2017)
- [x] [SimpleBaseline2D](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#simplebaseline2d-eccv-2018) (ECCV'2018)
- [x] [DSNT](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#dsnt-2018) (ArXiv'2021)
- [x] [HRNet](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#hrnet-cvpr-2019) (CVPR'2019)
- [x] [IPR](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#ipr-eccv-2018) (ECCV'2018)
- [x] [VideoPose3D](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#videopose3d-cvpr-2019) (CVPR'2019)
- [x] [HRNetv2](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#hrnetv2-tpami-2019) (TPAMI'2019)
- [x] [MSPN](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#mspn-arxiv-2019) (ArXiv'2019)
- [x] [SCNet](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#scnet-cvpr-2020) (CVPR'2020)
- [ ] [HigherHRNet](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#higherhrnet-cvpr-2020) (CVPR'2020)
- [x] [RSN](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#rsn-eccv-2020) (ECCV'2020)
- [ ] [InterNet](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#internet-eccv-2020) (ECCV'2020)
- [ ] [VoxelPose](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#voxelpose-eccv-2020) (ECCV'2020)
- [x] [LiteHRNet](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#litehrnet-cvpr-2021) (CVPR'2021)
- [x] [ViPNAS](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#vipnas-cvpr-2021) (CVPR'2021)
- [x] [Debias-IPR](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#debias-ipr-iccv-2021) (ICCV'2021)
- [x] [SimCC](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/algorithms.html#simcc-eccv-2022) (ECCV'2022)

</details>

<details close>
<summary><b>支持的技术</b></summary>

- [x] [FPN](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#fpn-cvpr-2017) (CVPR'2017)
- [x] [FP16](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#fp16-arxiv-2017) (ArXiv'2017)
- [x] [Wingloss](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#wingloss-cvpr-2018) (CVPR'2018)
- [x] [AdaptiveWingloss](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#adaptivewingloss-iccv-2019) (ICCV'2019)
- [x] [DarkPose](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#darkpose-cvpr-2020) (CVPR'2020)
- [x] [UDP](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#udp-cvpr-2020) (CVPR'2020)
- [x] [Albumentations](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#albumentations-information-2020) (Information'2020)
- [x] [SoftWingloss](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#softwingloss-tip-2021) (TIP'2021)
- [x] [RLE](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/techniques.html#rle-iccv-2021) (ICCV'2021)

</details>

<details close>
<summary><b>支持的数据集</b></summary>

- [x] [AFLW](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#aflw-iccvw-2011) \[[主页](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)\] (ICCVW'2011)
- [x] [sub-JHMDB](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#jhmdb-iccv-2013) \[[主页](http://jhmdb.is.tue.mpg.de/dataset)\] (ICCV'2013)
- [x] [COFW](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#cofw-iccv-2013) \[[主页](http://www.vision.caltech.edu/xpburgos/ICCV13/)\] (ICCV'2013)
- [x] [MPII](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#mpii-cvpr-2014) \[[主页](http://human-pose.mpi-inf.mpg.de/)\] (CVPR'2014)
- [x] [Human3.6M](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#human3-6m-tpami-2014) \[[主页](http://vision.imar.ro/human3.6m/description.php)\] (TPAMI'2014)
- [x] [COCO](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#coco-eccv-2014) \[[主页](http://cocodataset.org/)\] (ECCV'2014)
- [x] [CMU Panoptic](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#cmu-panoptic-iccv-2015) (ICCV'2015)
- [x] [DeepFashion](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#deepfashion-cvpr-2016) \[[主页](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)\] (CVPR'2016)
- [x] [300W](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#300w-imavis-2016) \[[主页](https://ibug.doc.ic.ac.uk/resources/300-W/)\] (IMAVIS'2016)
- [x] [RHD](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#rhd-iccv-2017) \[[主页](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)\] (ICCV'2017)
- [x] [CMU Panoptic](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#cmu-panoptic-iccv-2015) \[[主页](http://domedb.perception.cs.cmu.edu/)\] (ICCV'2015)
- [x] [AI Challenger](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#ai-challenger-arxiv-2017) \[[主页](https://github.com/AIChallenger/AI_Challenger_2017)\] (ArXiv'2017)
- [x] [MHP](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#mhp-acm-mm-2018) \[[主页](https://lv-mhp.github.io/dataset)\] (ACM MM'2018)
- [x] [WFLW](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#wflw-cvpr-2018) \[[主页](https://wywu.github.io/projects/LAB/WFLW.html)\] (CVPR'2018)
- [x] [PoseTrack18](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#posetrack18-cvpr-2018) \[[主页](https://posetrack.net/users/download.php)\] (CVPR'2018)
- [x] [OCHuman](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#ochuman-cvpr-2019) \[[主页](https://github.com/liruilong940607/OCHumanApi)\] (CVPR'2019)
- [x] [CrowdPose](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#crowdpose-cvpr-2019) \[[主页](https://github.com/Jeff-sjtu/CrowdPose)\] (CVPR'2019)
- [x] [MPII-TRB](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#mpii-trb-iccv-2019) \[[主页](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)\] (ICCV'2019)
- [x] [FreiHand](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#freihand-iccv-2019) \[[主页](https://lmb.informatik.uni-freiburg.de/projects/freihand/)\] (ICCV'2019)
- [x] [Animal-Pose](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#animal-pose-iccv-2019) \[[主页](https://sites.google.com/view/animal-pose/)\] (ICCV'2019)
- [x] [OneHand10K](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#onehand10k-tcsvt-2019) \[[主页](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)\] (TCSVT'2019)
- [x] [Vinegar Fly](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#vinegar-fly-nature-methods-2019) \[[主页](https://github.com/jgraving/DeepPoseKit-Data)\] (Nature Methods'2019)
- [x] [Desert Locust](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#desert-locust-elife-2019) \[[主页](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [Grévy’s Zebra](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#grevys-zebra-elife-2019) \[[主页](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [ATRW](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#atrw-acm-mm-2020) \[[主页](https://cvwc2019.github.io/challenge.html)\] (ACM MM'2020)
- [x] [Halpe](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#halpe-cvpr-2020) \[[主页](https://github.com/Fang-Haoshu/Halpe-FullBody/)\] (CVPR'2020)
- [x] [COCO-WholeBody](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#coco-wholebody-eccv-2020) \[[主页](https://github.com/jin-s13/COCO-WholeBody/)\] (ECCV'2020)
- [x] [MacaquePose](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#macaquepose-biorxiv-2020) \[[主页](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html)\] (bioRxiv'2020)
- [x] [InterHand2.6M](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#interhand2-6m-eccv-2020) \[[主页](https://mks0601.github.io/InterHand2.6M/)\] (ECCV'2020)
- [x] [AP-10K](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#ap-10k-neurips-2021) \[[主页](https://github.com/AlexTheBad/AP-10K)\] (NeurIPS'2021)
- [x] [Horse-10](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#horse-10-wacv-2021) \[[主页](http://www.mackenziemathislab.org/horse10)\] (WACV'2021)
- [x] [Human-Art](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#human-art-cvpr-2023) \[[主页](https://idea-research.github.io/HumanArt/)\] (CVPR'2023)
- [x] [LaPa](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/datasets.html#lapa-aaai-2020) \[[主页](https://github.com/JDAI-CV/lapa-dataset)\] (AAAI'2020)

</details>

<details close>
<summary><b>支持的骨干网络</b></summary>

- [x] [AlexNet](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#alexnet-neurips-2012) (NeurIPS'2012)
- [x] [VGG](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#vgg-iclr-2015) (ICLR'2015)
- [x] [ResNet](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#resnet-cvpr-2016) (CVPR'2016)
- [x] [ResNext](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#resnext-cvpr-2017) (CVPR'2017)
- [x] [SEResNet](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#seresnet-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV1](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#shufflenetv1-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV2](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#shufflenetv2-eccv-2018) (ECCV'2018)
- [x] [MobilenetV2](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#mobilenetv2-cvpr-2018) (CVPR'2018)
- [x] [ResNetV1D](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#resnetv1d-cvpr-2019) (CVPR'2019)
- [x] [ResNeSt](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#resnest-arxiv-2020) (ArXiv'2020)
- [x] [Swin](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#swin-cvpr-2021) (CVPR'2021)
- [x] [HRFormer](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#hrformer-nips-2021) (NIPS'2021)
- [x] [PVT](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#pvt-iccv-2021) (ICCV'2021)
- [x] [PVTV2](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo_papers/backbones.html#pvtv2-cvmj-2022) (CVMJ'2022)

</details>

### 模型需求

我们将跟进学界的最新进展，并支持更多算法和框架。如果您对 MMPose 有任何功能需求，请随时在 [MMPose Roadmap](https://github.com/open-mmlab/mmpose/issues/2258) 中留言。

## 参与贡献

我们非常欢迎用户对于 MMPose 做出的任何贡献，可以参考 [贡献指南](https://mmpose.readthedocs.io/zh_CN/latest/contribution_guide.html) 文件了解更多细节。

## 致谢

MMPose 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。
我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果您觉得 MMPose 对您的研究有所帮助，请考虑引用它：

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

## 许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源协议。

## OpenMMLab的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab 深度学习预训练工具箱
- [MMagic](https://github.com/open-mmlab/mmagic): OpenMMLab 新一代人工智能内容生成（AIGC）工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MIM](https://github.com/open-mmlab/mim): OpenMMlab 项目、算法、模型的统一入口
- [Playground](https://github.com/open-mmlab/playground): 收集和展示 OpenMMLab 相关的前沿、有趣的社区项目

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，联络 OpenMMLab [官方微信小助手](https://user-images.githubusercontent.com/25839884/205872898-e2e6009d-c6bb-4d27-8d07-117e697a3da8.jpg)或加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=K0QI8ByU)

<div align="center">
<img src="https://user-images.githubusercontent.com/25839884/205870927-39f4946d-8751-4219-a4c0-740117558fd7.jpg" height="400"><img src="https://user-images.githubusercontent.com/25839884/205872898-e2e6009d-c6bb-4d27-8d07-117e697a3da8.jpg" height="400"><img src="https://user-images.githubusercontent.com/25839884/203904835-62392033-02d4-4c73-a68c-c9e4c1e2b07f.jpg" height="400">
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
