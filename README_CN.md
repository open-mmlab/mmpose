<div align="center">
    <img src="resources/mmpose-logo.png" width="400"/>
</div>

## Introduction

[English](./README.md) | 简体中文

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://badge.fury.io/py/mmpose.svg)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

MMPose 是一款基于 PyTorch 的姿态分析的开源工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.3 以上**的版本

<div align="center">
    <img src="demo/resources/demo_coco.gif" width="600px" alt><br>
    COCO 17关键点 多人姿态估计
</div>
<div align="center">
<img src="https://user-images.githubusercontent.com/9464825/95552839-00a61080-0a40-11eb-818c-b8dad7307217.gif" width="600px" alt><br>

133关键点-多人全身姿态估计 ([高清完整版](https://www.youtube.com/watch?v=pIJpQg8mXUU))

</div>
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/11788150/114201893-4446ec00-9989-11eb-808b-5718c47c7b23.gif" width="600px" alt><br>
    2D 动物姿态估计
</div>

### 主要特性

- **支持多种人体姿态分析相关任务**

  MMPose 支持当前学界广泛关注的主流姿态分析任务：主要包括 2D多人姿态估计、2D手部姿态估计、2D人脸关键点检测、133关键点的全身人体姿态估计、3D人体形状恢复、服饰关键点检测、动物关键点检测等。
  具体请参考 [功能演示](demo/README.md)。

- **更高的精度和更快的速度**

  MMPose 复现了多种学界最先进的人体姿态分析模型，包括“自顶向下”和“自底向上”两大类算法。MMPose 相比于其他主流的代码库，具有更高的模型精度和训练速度。
  具体请参考 [基准测试](docs/benchmark.md)。

- **支持多样的数据集**

  MMPose 支持了很多主流数据集的准备和构建，如 COCO、 MPII 等。 具体请参考 [数据集准备](docs/data_preparation.md)。

- **模块化设计**

  MMPose 将统一的人体姿态分析框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的人体姿态分析模型。

- **详尽的单元测试和文档**

  MMPose 提供了详尽的说明文档，API 接口说明，全面的单元测试，以供社区参考。

## [模型库](https://mmpose.readthedocs.io/en/latest/modelzoo.html)

支持的算法:

<details open>
<summary>(点击收起)</summary>

- [x] [DeepPose](docs/papers/algorithms/deeppose.md) (CVPR'2014)
- [x] [CPM](docs/papers/backbones/cpm.md) (CVPR'2016)
- [x] [Hourglass](docs/papers/backbones/hourglass.md) (ECCV'2016)
- [x] [MSPN](docs/papers/backbones/mspn.md) (ArXiv'2019)
- [x] [RSN](docs/papers/backbones/rsn.md) (ECCV'2020)
- [x] [SimpleBaseline2D](docs/papers/algorithms/simplebaseline2d.md) (ECCV'2018)
- [x] [HRNet](docs/papers/backbones/hrnet.md) (CVPR'2019)
- [x] [HRNetv2](docs/papers/backbones/hrnetv2.md) (TPAMI'2019)
- [x] [SCNet](docs/papers/backbones/scnet.md) (CVPR'2020)
- [x] [Associative Embedding](docs/papers/algorithms/associative_embedding.md) (NeurIPS'2017)
- [x] [HigherHRNet](docs/papers/backbones/higherhrnet.md) (CVPR'2020)
- [x] [HMR](docs/papers/algorithms/hmr.md) (CVPR'2018)
- [x] [SimpleBaseline3D](docs/papers/algorithms/simplebaseline3d.md) (ICCV'2017)
- [x] [InterNet](docs/papers/algorithms/internet.md) (ECCV'2020)
- [x] [VideoPose3D](docs/papers/algorithms/videopose3d.md) (CVPR'2019)

</details>

支持的技术:

<details open>
<summary>(click to collapse)</summary>

- [x] [Wingloss](docs/papers/techniques/wingloss.md) (CVPR'2018)
- [x] [DarkPose](docs/papers/techniques/dark.md) (CVPR'2020)
- [x] [UDP](docs/papers/techniques/udp.md) (CVPR'2020)
- [x] [FP16](docs/papers/techniques/fp16.md) (ArXiv'2017)
- [x] [Albumentations](docs/papers/techniques/albumentations.md) (Information'2020)

</details>

支持的 [数据集](https://mmpose.readthedocs.io/en/latest/datasets.html):

<details open>
<summary>(点击收起)</summary>

- [x] [COCO](docs/papers/datasets/coco.md) \[[homepage](http://cocodataset.org/)\] (ECCV'2014)
- [x] [COCO-WholeBody](docs/papers/datasets/coco_wholebody.md) \[[homepage](https://github.com/jin-s13/COCO-WholeBody/)\] (ECCV'2020)
- [x] [MPII](docs/papers/datasets/mpii.md) \[[homepage](http://human-pose.mpi-inf.mpg.de/)\] (CVPR'2014)
- [x] [MPII-TRB](docs/papers/datasets/mpii_trb.md) \[[homepage](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)\] (ICCV'2019)
- [x] [AI Challenger](docs/papers/datasets/aic.md) \[[homepage](https://github.com/AIChallenger/AI_Challenger_2017)\] (ArXiv'2017)
- [x] [OCHuman](docs/papers/datasets/ochuman.md) \[[homepage](https://github.com/liruilong940607/OCHumanApi)\] (CVPR'2019)
- [x] [CrowdPose](docs/papers/datasets/crowdpose.md) \[[homepage](https://github.com/Jeff-sjtu/CrowdPose)\] (CVPR'2019)
- [x] [PoseTrack18](docs/papers/datasets/posetrack18.md) \[[homepage](https://posetrack.net/users/download.php)\] (CVPR'2018)
- [x] [MHP](docs/papers/datasets/mhp.md) \[[homepage](https://lv-mhp.github.io/dataset)\] (ACM MM'2018)
- [x] [sub-JHMDB](docs/papers/datasets/jhmdb.md) \[[homepage](http://jhmdb.is.tue.mpg.de/dataset)\] (ICCV'2013)
- [x] [Human3.6M](docs/papers/datasets/h36m.md) \[[homepage](http://vision.imar.ro/human3.6m/description.php)\] (TPAMI'2014)
- [x] [300W](docs/papers/datasets/300w.md) \[[homepage](https://ibug.doc.ic.ac.uk/resources/300-W/)\] (IMAVIS'2016)
- [x] [WFLW](docs/papers/datasets/wflw.md) \[[homepage](https://wywu.github.io/projects/LAB/WFLW.html)\] (CVPR'2018)
- [x] [AFLW](docs/papers/datasets/aflw.md) \[[homepage](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)\] (ICCVW'2011)
- [x] [COFW](docs/papers/datasets/cofw.md) \[[homepage](http://www.vision.caltech.edu/xpburgos/ICCV13/)\] (ICCV'2013)
- [x] [OneHand10K](docs/papers/datasets/onehand10k.md) \[[homepage](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)\] (TCSVT'2019)
- [x] [FreiHand](docs/papers/datasets/freihand.md) \[[homepage](https://lmb.informatik.uni-freiburg.de/projects/freihand/)\] (ICCV'2019)
- [x] [RHD](docs/papers/datasets/rhd.md) \[[homepage](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)\] (ICCV'2017)
- [x] [CMU Panoptic HandDB](docs/papers/datasets/panoptic.md) \[[homepage](http://domedb.perception.cs.cmu.edu/handdb.html)\] (CVPR'2017)
- [x] [InterHand2.6M](docs/papers/datasets/interhand.md) \[[homepage](https://mks0601.github.io/InterHand2.6M/)\] (ECCV'2020)
- [x] [DeepFashion](docs/papers/datasets/deepfashion.md) \[[homepage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)\] (CVPR'2016)
- [x] [Animal-Pose](docs/papers/datasets/animalpose.md) \[[homepage](https://sites.google.com/view/animal-pose/)\] (ICCV'2019)
- [x] [Horse-10](docs/papers/datasets/horse10.md) \[[homepage](http://www.mackenziemathislab.org/horse10)\] (WACV'2021)
- [x] [MacaquePose](docs/papers/datasets/macaque.md) \[[homepage](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html)\] (bioRxiv'2020)
- [x] [Vinegar Fly](docs/papers/datasets/fly.md) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Nature Methods'2019)
- [x] [Desert Locust](docs/papers/datasets/locust.md) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [Grévy’s Zebra](docs/papers/datasets/zebra.md) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [ATRW](docs/papers/datasets/atrw.md) \[[homepage](https://cvwc2019.github.io/challenge.html)\] (ACM MM'2020)

</details>

支持的骨干网络:

<details>
<summary>(点击打开)</summary>

- [x] [AlexNet](docs/papers/backbones/alexnet.md) (NeurIPS'2012)
- [x] [VGG](docs/papers/backbones/vgg.md) (ICLR'2015)
- [x] [ResNet](docs/papers/backbones/resnet.md) (CVPR'2016)
- [x] [ResNetV1D](docs/papers/backbones/resnetv1d.md) (CVPR'2019)
- [x] [ResNeSt](docs/papers/backbones/resnest.md) (ArXiv'2020)
- [x] [ResNext](docs/papers/backbones/resnext.md) (CVPR'2017)
- [x] [SEResNet](docs/papers/backbones/seresnet.md) (CVPR'2018)
- [x] [ShufflenetV1](docs/papers/backbones/shufflenetv1.md) (CVPR'2018)
- [x] [ShufflenetV2](docs/papers/backbones/shufflenetv2.md) (ECCV'2018)
- [x] [MobilenetV2](docs/papers/backbones/mobilenetv2.md) (CVPR'2018)

</details>

各个模型的结果和设置都可以在对应的 config（配置）目录下的 *README.md* 中查看。
整体的概况也可也在 [模型库](https://mmpose.readthedocs.io/en/latest/recognition_models.html) 页面中查看。

我们将跟进学界的最新进展，并支持更多算法和框架。如果您对 MMPose 有任何功能需求，请随时在 [问题](https://github.com/open-mmlab/mmpose/issues/9) 中留言。

## 基准测试

在主流的 COCO 姿态估计数据集上，进行基准测试。结果展示 MMPose 框架 具有更高的精度和训练速度。

| 骨干模型   | 输入分辨率 | MMPose (s/iter) | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) (s/iter) | MMPose (mAP) | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) (mAP) |
| :--------- | :--------: | :-------------: | :------------------------------------------------------------------------------: | :----------: | :---------------------------------------------------------------------------: |
| resnet_50  |  256x192   |    **0.28**     |                                       0.64                                       |  **0.718**   |                                     0.704                                     |
| resnet_50  |  384x288   |    **0.81**     |                                       1.24                                       |  **0.731**   |                                     0.722                                     |
| resnet_101 |  256x192   |    **0.36**     |                                       0.84                                       |  **0.726**   |                                     0.714                                     |
| resnet_101 |  384x288   |    **0.79**     |                                       1.53                                       |  **0.748**   |                                     0.736                                     |
| resnet_152 |  256x192   |    **0.49**     |                                       1.00                                       |  **0.735**   |                                     0.720                                     |
| resnet_152 |  384x288   |    **0.96**     |                                       1.65                                       |  **0.750**   |                                     0.743                                     |
| hrnet_w32  |  256x192   |    **0.54**     |                                       1.31                                       |  **0.746**   |                                     0.744                                     |
| hrnet_w32  |  384x288   |    **0.76**     |                                       2.00                                       |  **0.760**   |                                     0.758                                     |
| hrnet_w48  |  256x192   |    **0.66**     |                                       1.55                                       |  **0.756**   |                                     0.751                                     |
| hrnet_w48  |  384x288   |    **1.23**     |                                       2.20                                       |  **0.767**   |                                     0.763                                     |

更多详情可见 [基准测试](docs/benchmark.md)。

## 安装

请参考 [安装指南](docs/install.md) 进行安装。

## 数据准备

请参考 [data_preparation.md](docs/data_preparation.md) 进行数据集准备。

## 教程

请参考 [getting_started.md](docs/getting_started.md) 了解 MMPose 的基本使用。
MMPose 也提供了其他更详细的教程:

- [如何编写配置文件](docs/tutorials/0_config.md)
- [如何微调模型](docs/tutorials/1_finetune.md)
- [如何增加新数据集](docs/tutorials/2_new_dataset.md)
- [如何设计数据处理流程](docs/tutorials/3_data_pipeline.md)
- [如何增加新模块](docs/tutorials/4_new_modules.md)
- [如何导出模型为 onnx 格式](docs/tutorials/5_export_model.md)
- [如何自定义模型运行参数](docs/tutorials/6_customize_runtime.md)

## 常见问题

请参考 [FAQ](docs/faq.md) 了解其他用户的常见问题。

## 许可

该项目采用 [Apache 2.0 license](LICENSE) 开源协议。

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

## 参与贡献

我们非常欢迎用户对于 MMPose 做出的任何贡献，可以参考 [CONTRIBUTION.md](.github/CONTRIBUTING.md) 文件了解更多细节。

## 致谢

MMPose 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。
我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## OpenMMLab的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 新一代生成模型工具箱

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="docs/imgs/zhihu_qrcode.jpg" height="400" />  <img src="docs/imgs/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
