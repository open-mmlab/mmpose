<div align="center">
    <img src="resources/mmpose-logo.png" width="400"/>
</div>

## Introduction

[English](./README.md) | 简体中文

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/zh_CN/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://img.shields.io/pypi/v/mmpose)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

MMPose 是一款基于 PyTorch 的姿态分析的开源工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.5 以上**的版本。

https://user-images.githubusercontent.com/15977946/124654387-0fd3c500-ded1-11eb-84f6-24eeddbf4d91.mp4

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

## [模型库](https://mmpose.readthedocs.io/zh_CN/latest/modelzoo.html)

支持的算法:

<details open>
<summary>(点击收起)</summary>

- [x] [DeepPose](https://mmpose.readthedocs.io/zh_CN/latest/papers/algorithms.html#deeppose-cvpr-2014) (CVPR'2014)
- [x] [CPM](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#cpm-cvpr-2016) (CVPR'2016)
- [x] [Hourglass](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#hourglass-eccv-2016) (ECCV'2016)
- [x] [MSPN](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#mspn-arxiv-2019) (ArXiv'2019)
- [x] [RSN](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#rsn-eccv-2020) (ECCV'2020)
- [x] [SimpleBaseline2D](https://mmpose.readthedocs.io/zh_CN/latest/papers/algorithms.html#simplebaseline2d-eccv-2018) (ECCV'2018)
- [x] [HRNet](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#hrnet-cvpr-2019) (CVPR'2019)
- [x] [HRNetv2](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#hrnetv2-tpami-2019) (TPAMI'2019)
- [x] [LiteHRNet](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#litehrnet-cvpr-2021) (CVPR'2021)
- [x] [SCNet](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#scnet-cvpr-2020) (CVPR'2020)
- [x] [Associative Embedding](https://mmpose.readthedocs.io/zh_CN/latest/papers/algorithms.html#associative-embedding-nips-2017) (NeurIPS'2017)
- [x] [HigherHRNet](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#higherhrnet-cvpr-2020) (CVPR'2020)
- [x] [HMR](https://mmpose.readthedocs.io/zh_CN/latest/papers/algorithms.html#hmr-cvpr-2018) (CVPR'2018)
- [x] [SimpleBaseline3D](https://mmpose.readthedocs.io/zh_CN/latest/papers/algorithms.html#simplebaseline3d-iccv-2017) (ICCV'2017)
- [x] [InterNet](https://mmpose.readthedocs.io/zh_CN/latest/papers/algorithms.html#internet-eccv-2020) (ECCV'2020)
- [x] [VideoPose3D](https://mmpose.readthedocs.io/zh_CN/latest/papers/algorithms.html#videopose3d-cvpr-2019) (CVPR'2019)
- [x] [ViPNAS](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#vipnas-cvpr-2021) (CVPR'2021)

</details>

支持的技术:

<details open>
<summary>(click to collapse)</summary>

- [x] [Wingloss](https://mmpose.readthedocs.io/zh_CN/latest/papers/techniques.html#wingloss-cvpr-2018) (CVPR'2018)
- [x] [DarkPose](https://mmpose.readthedocs.io/zh_CN/latest/papers/techniques.html#darkpose-cvpr-2020) (CVPR'2020)
- [x] [UDP](https://mmpose.readthedocs.io/zh_CN/latest/papers/techniques.html#udp-cvpr-2020) (CVPR'2020)
- [x] [FP16](https://mmpose.readthedocs.io/zh_CN/latest/papers/techniques.html#fp16-arxiv-2017) (ArXiv'2017)
- [x] [Albumentations](https://mmpose.readthedocs.io/zh_CN/latest/papers/techniques.html#albumentations-information-2020) (Information'2020)

</details>

支持的 [数据集](https://mmpose.readthedocs.io/zh_CN/latest/datasets.html):

<details open>
<summary>(点击收起)</summary>

- [x] [COCO](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#coco-eccv-2014) \[[homepage](http://cocodataset.org/)\] (ECCV'2014)
- [x] [COCO-WholeBody](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#coco-wholebody-eccv-2020) \[[homepage](https://github.com/jin-s13/COCO-WholeBody/)\] (ECCV'2020)
- [x] [Halpe](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#halpe-cvpr-2020) \[[homepage](https://github.com/Fang-Haoshu/Halpe-FullBody/)\] (CVPR'2020)
- [x] [MPII](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#mpii-cvpr-2014) \[[homepage](http://human-pose.mpi-inf.mpg.de/)\] (CVPR'2014)
- [x] [MPII-TRB](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#mpii-trb-iccv-2019) \[[homepage](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)\] (ICCV'2019)
- [x] [AI Challenger](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#ai-challenger-arxiv-2017) \[[homepage](https://github.com/AIChallenger/AI_Challenger_2017)\] (ArXiv'2017)
- [x] [OCHuman](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#ochuman-cvpr-2019) \[[homepage](https://github.com/liruilong940607/OCHumanApi)\] (CVPR'2019)
- [x] [CrowdPose](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#crowdpose-cvpr-2019) \[[homepage](https://github.com/Jeff-sjtu/CrowdPose)\] (CVPR'2019)
- [x] [PoseTrack18](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#posetrack18-cvpr-2018) \[[homepage](https://posetrack.net/users/download.php)\] (CVPR'2018)
- [x] [MHP](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#mhp-acm-mm-2018) \[[homepage](https://lv-mhp.github.io/dataset)\] (ACM MM'2018)
- [x] [sub-JHMDB](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#jhmdb-iccv-2013) \[[homepage](http://jhmdb.is.tue.mpg.de/dataset)\] (ICCV'2013)
- [x] [Human3.6M](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#human3-6m-tpami-2014) \[[homepage](http://vision.imar.ro/human3.6m/description.php)\] (TPAMI'2014)
- [x] [300W](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#300w-imavis-2016) \[[homepage](https://ibug.doc.ic.ac.uk/resources/300-W/)\] (IMAVIS'2016)
- [x] [WFLW](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#wflw-cvpr-2018) \[[homepage](https://wywu.github.io/projects/LAB/WFLW.html)\] (CVPR'2018)
- [x] [AFLW](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#aflw-iccvw-2011) \[[homepage](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)\] (ICCVW'2011)
- [x] [COFW](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#cofw-iccv-2013) \[[homepage](http://www.vision.caltech.edu/xpburgos/ICCV13/)\] (ICCV'2013)
- [x] [OneHand10K](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#onehand10k-tcsvt-2019) \[[homepage](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)\] (TCSVT'2019)
- [x] [FreiHand](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#freihand-iccv-2019) \[[homepage](https://lmb.informatik.uni-freiburg.de/projects/freihand/)\] (ICCV'2019)
- [x] [RHD](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#rhd-iccv-2017) \[[homepage](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)\] (ICCV'2017)
- [x] [CMU Panoptic HandDB](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#cmu-panoptic-handdb-cvpr-2017) \[[homepage](http://domedb.perception.cs.cmu.edu/handdb.html)\] (CVPR'2017)
- [x] [InterHand2.6M](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#interhand2-6m-eccv-2020) \[[homepage](https://mks0601.github.io/InterHand2.6M/)\] (ECCV'2020)
- [x] [DeepFashion](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#deepfashion-cvpr-2016) \[[homepage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)\] (CVPR'2016)
- [x] [Animal-Pose](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#animal-pose-iccv-2019) \[[homepage](https://sites.google.com/view/animal-pose/)\] (ICCV'2019)
- [x] [AP-10K](https://arxiv.org/abs/2108.12617) \[[homepage](https://github.com/AlexTheBad/AP-10K)\] (NeurIPS'2021)
- [x] [Horse-10](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#horse-10-wacv-2021) \[[homepage](http://www.mackenziemathislab.org/horse10)\] (WACV'2021)
- [x] [MacaquePose](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#macaquepose-biorxiv-2020) \[[homepage](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html)\] (bioRxiv'2020)
- [x] [Vinegar Fly](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#vinegar-fly-nature-methods-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Nature Methods'2019)
- [x] [Desert Locust](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#desert-locust-elife-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [Grévy’s Zebra](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#grevys-zebra-elife-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [ATRW](https://mmpose.readthedocs.io/zh_CN/latest/papers/datasets.html#atrw-acm-mm-2020) \[[homepage](https://cvwc2019.github.io/challenge.html)\] (ACM MM'2020)

</details>

支持的骨干网络:

<details>
<summary>(点击打开)</summary>

- [x] [AlexNet](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#alexnet-neurips-2012) (NeurIPS'2012)
- [x] [VGG](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#vgg-iclr-2015) (ICLR'2015)
- [x] [ResNet](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#resnet-cvpr-2016) (CVPR'2016)
- [x] [ResNetV1D](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#resnetv1d-cvpr-2019) (CVPR'2019)
- [x] [ResNeSt](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#resnest-arxiv-2020) (ArXiv'2020)
- [x] [ResNext](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#resnext-cvpr-2017) (CVPR'2017)
- [x] [SEResNet](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#seresnet-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV1](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#shufflenetv1-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV2](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#shufflenetv2-eccv-2018) (ECCV'2018)
- [x] [MobilenetV2](https://mmpose.readthedocs.io/zh_CN/latest/papers/backbones.html#mobilenetv2-cvpr-2018) (CVPR'2018)

</details>

各个模型的结果和设置都可以在对应的 config（配置）目录下的 *README.md* 中查看。
整体的概况也可也在 [模型库](https://mmpose.readthedocs.io/zh_CN/latest/recognition_models.html) 页面中查看。

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
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 新一代生成模型工具箱

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=GJP18SjI)

<div align="center">
<img src="docs/imgs/zhihu_qrcode.jpg" height="400" />  <img src="docs/imgs/qq_group2_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
