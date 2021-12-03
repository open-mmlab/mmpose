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

## 推理速度总结

这里总结了 MMPose 中主要模型的复杂度信息和推理速度，包括模型的计算复杂度、参数数量，以及以不同的批处理大小在 CPU 和 GPU 上的推理速度。

<details open>
<summary>(点击收起)</summary>

| 算法 | 模型 | 配置文件 | 输入分辨率 | 全类别平均正确率 | 浮点数运算次数<br>(10亿) | 参数数量<br>(百万) | GPU 上的推理速度<br>(每秒处理的帧数)| GPU 上的推理速度<br>(每秒处理的帧数, 批处理大小为10) | CPU 上的推理速度<br>(每秒处理的帧数) | CPU 上的推理速度<br>(每秒处理的帧数, 批处理大小为10) |
| :--- | :---------------: | :-----------------: |:--------------------: | :----------------------------: | :-----------------: | :---------------: |:--------------------: | :----------------------------: | :-----------------: | :-----------------: |
| topdown_heatmap | Alexnet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/alexnet_coco_256x192.py) | (3, 192, 256) | 0.397 | 1.42 | 5.62 | 229.21 ± 16.91 | 33.52 ± 1.14 | 13.92 ± 0.60 | 1.38 ± 0.02 |
| topdown_heatmap | CPM | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/cpm_coco_256x192.py) | (3, 192, 256) | 0.623 | 63.81 | 31.3 | 11.35 ± 0.22 | 3.87 ± 0.07 | 0.31 ± 0.01 | 0.03 ± 0.00 |
| topdown_heatmap | CPM | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/cpm_coco_384x288.py) | (3, 288, 384) | 0.65 | 143.57 | 31.3 | 7.09 ± 0.14 | 2.10 ± 0.05 | 0.14 ± 0.00 | 0.01 ± 0.00 |
| topdown_heatmap | Hourglass | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_256x256.py) | (3, 256, 256) | 0.726 | 28.67 | 94.85 | 25.50 ± 1.68 | 3.99 ± 0.07 | 0.92 ± 0.03 | 0.09 ± 0.00 |
| topdown_heatmap | Hourglass | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_384x384.py) | (3, 384, 384) | 0.746 | 64.5 | 94.85 | 14.74 ± 0.8 | 1.86 ± 0.06 | 0.43 ± 0.03 | 0.04 ± 0.00 |
| topdown_heatmap | HRNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py) | (3, 192, 256) | 0.746 | 7.7 | 28.54 | 22.73 ± 1.12 | 6.60 ± 0.14 | 2.73 ± 0.11 | 0.32 ± 0.00 |
| topdown_heatmap | HRNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_384x288.py) | (3, 288, 384) | 0.76 | 17.33 | 28.54 | 22.78 ± 1.21 | 3.28 ± 0.08 | 1.35 ± 0.05 | 0.14 ± 0.00 |
| topdown_heatmap | HRNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py) | (3, 192, 256) | 0.756 | 15.77 | 63.6 | 22.01 ± 1.10 | 3.74 ± 0.10 | 1.46 ± 0.05 | 0.16 ± 0.00 |
| topdown_heatmap | HRNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py) | (3, 288, 384) | 0.767 | 35.48 | 63.6 | 15.03 ± 1.03 | 1.80 ± 0.03 | 0.68 ± 0.02 | 0.07 ± 0.00 |
| topdown_heatmap | LiteHRNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_256x192.py) | (3, 192, 256) | 0.675 | 0.42 | 1.76 | 11.86 ± 0.38 | 9.77 ± 0.23 | 5.84 ± 0.39 | 0.80 ± 0.00 |
| topdown_heatmap | LiteHRNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_384x288.py) | (3, 288, 384) | 0.7 | 0.95 | 1.76 | 11.52 ± 0.39 | 5.18 ± 0.11 | 3.45 ± 0.22 | 0.37 ± 0.00 |
| topdown_heatmap | MobilenetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_256x192.py) | (3, 192, 256) | 0.646 | 1.59 | 9.57 | 91.82 ± 10.98 | 17.85 ± 0.32 | 10.44 ± 0.80 | 1.05 ± 0.01 |
| topdown_heatmap | MobilenetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_384x288.py) | (3, 288, 384) | 0.673 | 3.57 | 9.57 | 71.27 ± 6.82 | 8.00 ± 0.15  | 5.01 ± 0.32 | 0.46 ± 0.00 |
| topdown_heatmap | MSPN | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mspn50_coco_256x192.py) | (3, 192, 256) | 0.723 | 5.11 | 25.11 | 59.65 ± 3.74 | 9.51 ± 0.15  | 3.98 ± 0.21 | 0.43 ± 0.00 |
| topdown_heatmap | MSPN | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/2xmspn50_coco_256x192.py) | (3, 192, 256) | 0.754 | 11.35 | 56.8 | 30.64 ± 2.61 | 4.74 ± 0.12 | 1.85 ± 0.08 | 0.20 ± 0.00 |
| topdown_heatmap | MSPN | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/3xmspn50_coco_256x192.py) | (3, 192, 256) | 0.758 | 17.59 | 88.49 | 20.90 ± 1.82 | 3.22 ± 0.08 | 1.23 ± 0.04 | 0.13 ± 0.00 |
| topdown_heatmap | MSPN | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/4xmspn50_coco_256x192.py) | (3, 192, 256) | 0.764 | 23.82 | 120.18 | 15.79 ± 1.14  | 2.45 ± 0.05 | 0.90 ± 0.03 | 0.10 ± 0.00 |
| topdown_heatmap | ResNest | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest50_coco_256x192.py) | (3, 192, 256) | 0.721 | 6.73 | 35.93 | 48.36 ± 4.12 | 7.48 ± 0.13 | 3.00 ± 0.13 | 0.33 ± 0.00 |
| topdown_heatmap | ResNest | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest50_coco_384x288.py) | (3, 288, 384) | 0.737 | 15.14 | 35.93 | 30.30 ± 2.30 | 3.62 ± 0.09 | 1.43 ± 0.05 | 0.13 ± 0.00 |
| topdown_heatmap | ResNest | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest101_coco_256x192.py) | (3, 192, 256) | 0.725 | 10.38 | 56.61 | 29.21 ± 1.98 | 5.30 ± 0.12 | 2.01 ± 0.08 | 0.22 ± 0.00 |
| topdown_heatmap | ResNest | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest101_coco_384x288.py) | (3, 288, 384) | 0.746 | 23.36 | 56.61 | 19.02 ± 1.40 | 2.59 ± 0.05  | 0.97 ± 0.03 | 0.09 ± 0.00 |
| topdown_heatmap | ResNest | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest200_coco_256x192.py) | (3, 192, 256) | 0.732 | 17.5 | 78.54 | 16.11 ± 0.71 | 3.29 ± 0.07  | 1.33 ± 0.02 | 0.14 ± 0.00 |
| topdown_heatmap | ResNest | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest200_coco_384x288.py) | (3, 288, 384) | 0.754 | 39.37 | 78.54 | 11.48 ± 0.68 | 1.58 ± 0.02 | 0.63 ± 0.01 | 0.06 ± 0.00 |
| topdown_heatmap | ResNest | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest269_coco_256x192.py) | (3, 192, 256) | 0.738 | 22.45 | 119.27 | 12.02 ± 0.47 | 2.60 ± 0.05 | 1.03 ± 0.01 | 0.11 ± 0.00 |
| topdown_heatmap | ResNest | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest269_coco_384x288.py) | (3, 288, 384) | 0.755 | 50.5 | 119.27 | 8.82 ± 0.42  | 1.24 ± 0.02 | 0.49 ± 0.01 | 0.05 ± 0.00 |
| topdown_heatmap | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py) | (3, 192, 256) | 0.718 | 5.46 | 34 | 64.23 ± 6.05 | 9.33 ± 0.21 | 4.00 ± 0.10 | 0.41 ± 0.00 |
| topdown_heatmap | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_384x288.py) | (3, 288, 384) | 0.731 | 12.29 | 34 | 36.78 ± 3.05 | 4.48 ± 0.12 | 1.92 ± 0.04 | 0.19 ± 0.00 |
| topdown_heatmap | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192.py) | (3, 192, 256) | 0.726 | 9.11 | 52.99 | 43.35 ± 4.36 | 6.44 ± 0.14 | 2.57 ± 0.05 | 0.27 ± 0.00 |
| topdown_heatmap | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_384x288.py) | (3, 288, 384) | 0.748 | 20.5 | 52.99 | 23.29 ± 1.83 | 3.12 ± 0.09 | 1.23 ± 0.03 | 0.11 ± 0.00 |
| topdown_heatmap | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192.py) | (3, 192, 256) | 0.735 | 12.77 | 68.64 | 32.31 ± 2.84 | 4.88 ± 0.17 | 1.89 ± 0.03 | 0.20 ± 0.00 |
| topdown_heatmap | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_384x288.py) | (3, 288, 384) | 0.75 | 28.73 | 68.64 | 17.32 ± 1.17 | 2.40 ± 0.04 | 0.91 ± 0.01 | 0.08 ± 0.00 |
| topdown_heatmap | ResNetV1d | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_256x192.py) | (3, 192, 256) | 0.722 | 5.7 | 34.02 | 63.44 ± 6.09 | 9.09 ± 0.10 | 3.82 ± 0.10 | 0.39 ± 0.00 |
| topdown_heatmap | ResNetV1d | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_384x288.py) | (3, 288, 384) | 0.73 | 12.82 | 34.02 | 36.21 ± 3.10 | 4.30 ± 0.12 | 1.82 ± 0.04 | 0.16 ± 0.00 |
| topdown_heatmap | ResNetV1d | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d101_coco_256x192.py) | (3, 192, 256) | 0.731 | 9.35 | 53.01 | 41.48 ± 3.76 | 6.33 ± 0.15 | 2.48 ± 0.05 | 0.26 ± 0.00 |
| topdown_heatmap | ResNetV1d | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d101_coco_384x288.py) | (3, 288, 384) | 0.748 | 21.04 | 53.01 | 23.49 ± 1.76 | 3.07 ± 0.07 | 1.19 ± 0.02 | 0.11 ± 0.00 |
| topdown_heatmap | ResNetV1d | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d152_coco_256x192.py) | (3, 192, 256) | 0.737 | 13.01 | 68.65 | 31.96 ± 2.87 | 4.69 ± 0.18 | 1.87 ± 0.02 | 0.19 ± 0.00 |
| topdown_heatmap | ResNetV1d | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d152_coco_384x288.py) | (3, 288, 384) | 0.752 | 29.26 | 68.65 | 17.31 ± 1.13 | 2.32 ± 0.04 | 0.88 ± 0.01 | 0.08 ± 0.00 |
| topdown_heatmap | ResNext | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext50_coco_256x192.py) | (3, 192, 256) | 0.714 | 5.61 | 33.47 | 48.34 ± 3.85 | 7.66 ± 0.13 | 3.71 ± 0.10 | 0.37 ± 0.00 |
| topdown_heatmap | ResNext | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext50_coco_384x288.py) | (3, 288, 384) | 0.724 | 12.62 | 33.47 | 30.66 ± 2.38 | 3.64 ± 0.11 | 1.73 ± 0.03 | 0.15 ± 0.00 |
| topdown_heatmap | ResNext | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext101_coco_256x192.py) | (3, 192, 256) | 0.726 | 9.29 | 52.62 | 27.33 ± 2.35 | 5.09 ± 0.13 | 2.45 ± 0.04 | 0.25 ± 0.00 |
| topdown_heatmap | ResNext | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext101_coco_384x288.py) | (3, 288, 384) | 0.743 | 20.91 | 52.62 | 18.19 ± 1.38  | 2.42 ± 0.04 | 1.15 ± 0.01 | 0.10 ± 0.00 |
| topdown_heatmap | ResNext | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext152_coco_256x192.py) | (3, 192, 256) | 0.73 | 12.98 | 68.39 | 19.61 ± 1.61 | 3.80 ± 0.13 | 1.83 ± 0.02 | 0.18 ± 0.00 |
| topdown_heatmap | ResNext | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext152_coco_384x288.py) | (3, 288, 384) | 0.742 | 29.21 | 68.39 | 13.14 ± 0.75 | 1.82 ± 0.03 | 0.85 ± 0.01 | 0.08 ± 0.00 |
| topdown_heatmap | RSN | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/rsn18_coco_256x192.py) | (3, 192, 256) | 0.704 | 2.27 | 9.14 | 47.80 ± 4.50 | 13.68 ± 0.25 | 6.70 ± 0.28 | 0.70 ± 0.00 |
| topdown_heatmap | RSN | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/rsn50_coco_256x192.py) | (3, 192, 256) | 0.723 | 4.11 | 19.33 | 27.22 ± 1.61 | 8.81 ± 0.13 | 3.98 ± 0.12 | 0.45 ± 0.00 |
| topdown_heatmap | RSN | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/2xrsn50_coco_256x192.py) | (3, 192, 256) | 0.745 | 8.29 | 39.26 | 13.88 ± 0.64 | 4.78 ± 0.13 | 2.02 ± 0.04 | 0.23 ± 0.00 |
| topdown_heatmap | RSN | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/3xrsn50_coco_256x192.py) | (3, 192, 256) | 0.75 | 12.47 | 59.2 | 9.40 ± 0.32 | 3.37 ± 0.09 | 1.34 ± 0.03 | 0.15 ± 0.00 |
| topdown_heatmap | SCNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet50_coco_256x192.py) | (3, 192, 256) | 0.728 | 5.31 | 34.01 | 40.76 ± 3.08 | 8.35 ± 0.19 | 3.82 ± 0.08 | 0.40 ± 0.00 |
| topdown_heatmap | SCNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet50_coco_384x288.py) | (3, 288, 384) | 0.751 | 11.94 | 34.01 | 32.61 ± 2.97 | 4.19 ± 0.10 | 1.85 ± 0.03 | 0.17 ± 0.00 |
| topdown_heatmap | SCNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet101_coco_256x192.py) | (3, 192, 256) | 0.733 | 8.51 | 53.01 | 24.28 ± 1.19 | 5.80 ± 0.13 | 2.49 ± 0.05 | 0.27 ± 0.00  |
| topdown_heatmap | SCNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet101_coco_384x288.py) | (3, 288, 384) | 0.752 | 19.14 | 53.01 | 20.43 ± 1.76 | 2.91 ± 0.06 | 1.23 ± 0.02 | 0.12 ± 0.00 |
| topdown_heatmap | SeresNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet50_coco_256x192.py) | (3, 192, 256) | 0.728 | 5.47 | 36.53 | 54.83 ± 4.94 | 8.80 ± 0.12 | 3.85 ± 0.10 | 0.40 ± 0.00 |
| topdown_heatmap | SeresNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet50_coco_384x288.py) | (3, 288, 384) | 0.748 | 12.3 | 36.53 | 33.00 ± 2.67 | 4.26 ± 0.12 | 1.86 ± 0.04 | 0.17 ± 0.00 |
| topdown_heatmap | SeresNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet101_coco_256x192.py) | (3, 192, 256) | 0.734 | 9.13 | 57.77 | 33.90 ± 2.65 | 6.01 ± 0.13 | 2.48 ± 0.05 | 0.26 ± 0.00 |
| topdown_heatmap | SeresNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet101_coco_384x288.py) | (3, 288, 384) | 0.753 | 20.53 | 57.77 | 20.57 ± 1.57 | 2.96 ± 0.07 | 1.20 ± 0.02 | 0.11 ± 0.00 |
| topdown_heatmap | SeresNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet152_coco_256x192.py) | (3, 192, 256) | 0.73 | 12.79 | 75.26 | 24.25 ± 1.95 | 4.45 ± 0.10 | 1.82 ± 0.02 | 0.19 ± 0.00 |
| topdown_heatmap | SeresNet | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet152_coco_384x288.py) | (3, 288, 384) | 0.753 | 28.76 | 75.26 | 15.11 ± 0.99  | 2.25 ± 0.04 | 0.88 ± 0.01 | 0.08 ± 0.00 |
| topdown_heatmap | ShuffleNetV1 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv1_coco_256x192.py) | (3, 192, 256) | 0.585 | 1.35 | 6.94 | 80.79 ± 8.95 | 21.91 ± 0.46 | 11.84 ± 0.59 | 1.25 ± 0.01 |
| topdown_heatmap | ShuffleNetV1 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv1_coco_384x288.py) | (3, 288, 384) | 0.622 | 3.05 | 6.94 | 63.45 ± 5.21 | 9.84 ± 0.10 | 6.01 ± 0.31 | 0.57 ± 0.00 |
| topdown_heatmap | ShuffleNetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_256x192.py) | (3, 192, 256) | 0.599 | 1.37 | 7.55 | 82.36 ± 7.30 | 22.68 ± 0.53 | 12.40 ± 0.66 | 1.34 ± 0.02 |
| topdown_heatmap | ShuffleNetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_384x288.py) | (3, 288, 384) | 0.636 | 3.08 | 7.55 | 63.63 ± 5.72 | 10.47 ± 0.16 | 6.32 ± 0.28 | 0.63 ± 0.01  |
| topdown_heatmap | VGG | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vgg16_bn_coco_256x192.py) | (3, 192, 256) | 0.698 | 16.22 | 18.92 | 51.91 ± 2.98 | 6.18 ± 0.13 | 1.64 ± 0.03 | 0.15 ± 0.00 |
| topdown_heatmap | VIPNAS + Res50 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py) | (3, 192, 256) | 0.711 | 1.49 | 7.29 | 34.88 ± 2.45 | 10.29 ± 0.13 | 6.51 ± 0.17 | 0.65 ± 0.00 |
| topdown_heatmap | VIPNAS + MobileNetV3 | [config](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_mbv3_coco_256x192.py) | (3, 192, 256) | 0.7 | 0.76 | 5.9 | 53.62 ± 6.59 | 11.54 ± 0.18 | 1.26 ± 0.02 | 0.13 ± 0.00 |
| Associative Embedding | HigherHRNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py) | (3, 512, 512) | 0.677 | 46.58 | 28.65 | 7.80 ± 0.67 | / | 0.28 ± 0.02 | / |
| Associative Embedding | HigherHRNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_640x640.py) | (3, 640, 640) | 0.686 | 72.77 | 28.65 | 5.30 ± 0.37 | / | 0.17 ± 0.01 | / |
| Associative Embedding | HigherHRNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py) | (3, 512, 512) | 0.686 | 96.17 | 63.83 | 4.55 ± 0.35 | / | 0.15 ± 0.01 | / |
| Associative Embedding | Hourglass | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hourglass_ae_coco_512x512.py) | (3, 512, 512) | 0.613 | 221.58 | 138.86 | 3.55 ± 0.24 | / | 0.08 ± 0.00 | / |
| Associative Embedding | HRNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py) | (3, 512, 512) | 0.654 | 41.1 | 28.54 | 8.93 ± 0.76 | / | 0.33 ± 0.02 | / |
| Associative Embedding | HRNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_coco_512x512.py) | (3, 512, 512) | 0.665 | 84.12 | 63.6 | 5.27 ± 0.43 | / | 0.18 ± 0.01 | / |
| Associative Embedding | MobilenetV2 | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/mobilenetv2_coco_512x512.py) | (3, 512, 512) | 0.38 | 8.54 | 9.57 | 21.24 ± 1.34 | / | 0.81 ± 0.06 | / |
| Associative Embedding | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_512x512.py) | (3, 512, 512) | 0.466 | 29.2 | 34 | 11.71 ± 0.97 | / | 0.41 ± 0.02 | / |
| Associative Embedding | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_640x640.py) | (3, 640, 640) | 0.479 | 45.62 | 34 | 8.20 ± 0.58 | / | 0.26 ± 0.02 | / |
| Associative Embedding | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res101_coco_512x512.py) | (3, 512, 512) | 0.554 | 48.67 | 53 | 8.26 ± 0.68 | / | 0.28 ± 0.02 | / |
| Associative Embedding | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res152_coco_512x512.py) | (3, 512, 512) | 0.595 | 68.17 | 68.64 | 6.25 ± 0.53 | / | 0.21 ± 0.01 | / |
| DeepPose | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py) | (3, 192, 256) | 0.526 | 4.04 | 23.58 | 82.20 ± 7.54 | / | 5.50 ± 0.18 | / |
| DeepPose | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res101_coco_256x192.py) | (3, 192, 256) | 0.56 | 7.69 | 42.57 | 48.93 ± 4.02 | / | 3.10 ± 0.07 | / |
| DeepPose | ResNet | [config](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res152_coco_256x192.py) | (3, 192, 256) | 0.583 | 11.34 | 58.21 | 35.06 ± 3.50 | / | 2.19 ± 0.04 | / |

</details>

更多关于模型推理速度的详细信息可见 [模型推理速度](docs_zh-CN/inference_speed_summary.md).

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
