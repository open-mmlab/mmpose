# RTMPose: Real-Time Multi-Person Pose Estimation toolkit based on MMPose

## Abstract

Recent studies on 2D pose estimation have achieved excellent performance on public benchmarks, yet its application in the industrial community still suffers from heavy model parameters and high latency.
In order to bridge this gap, we empirically study five aspects that affect the performance of multi-person pose estimation algorithms: paradigm, backbone network, localization algorithm, training strategy, and deployment inference, and present a high-performance real-time multi-person pose estimation framework, **RTMPose**, based on MMPose.
Our RTMPose-m achieves **75.8% AP** on COCO with **90+ FPS** on an Intel i7-11700 CPU and **400+ FPS** on an NVIDIA GTX 1660 Ti GPU, and RTMPose-l achieves **67.0% AP** on COCO-WholeBody with **130+ FPS**, outperforming existing open-source libraries.
To further evaluate RTMPose's capability in critical real-time applications, we also report the performance after deploying on the mobile device.

![rtmpose_intro](https://user-images.githubusercontent.com/13503330/219269619-935499e5-bdd9-49ea-8104-3c7796dbd862.png)

## 引用我们

Coming soon

## 简介

[English](README.md) | 简体中文

### 主要特性

- **高精度，低延迟**

  - t | COCO 68.5 AP | CPU: 300+ FPS / GPU: 940+ FPS
  - s | COCO 72.2 AP | CPU: 200+ FPS / GPU: 710+ FPS
  - m | COCO 75.8 AP | CPU: 90+ FPS / GPU: 430+ FPS
  - l | COCO 76.5 AP | CPU: 50+ FPS / GPU: 280+ FPS

- **易部署**

  - 详细的部署代码教程，手把手教你模型部署
  - MMDeploy 助力，支持多种部署后端: ONNX, TensorRT, ncnn, OpenVINO 等

- **为实际业务设计**

  提供多种 Pipeline 推理接口和 SDK: Python, C++, C# 等

## 最新进展

- 2023 年 2 月：发布 RTMPose。RTMPose-m 取得 COCO 验证集 75.8 mAP，推理速度达到 90+ FPS 。

## 社区共建

RTMPose 是一个长期优化迭代的项目，致力于业务场景下的高性能实时姿态估计算法的训练、优化和部署，因此我们十分期待来自社区的力量，欢迎分享不同业务场景中 RTMPose 的训练配置与技巧，助力更多的社区用户！

✨ ✨ ✨

- **如果你是 RTMPose 的新用户，我们热切希望你能参与[这份问卷](https://uua478.fanqier.cn/f/xxmynrki)/[Google Questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSfzwWr3eNlDzhU98qzk2Eph44Zio6hi5r0iSwfO9wSARkHdWg/viewform?usp=sf_link)，这对于我们的工作非常重要！**

✨ ✨ ✨

欢迎加入我们的社区交流群获得更多帮助：

- 微信用户群
- Discord Group: https://discord.gg/raweFPmdzG

## Pipeline 性能

**说明**

- Pipeline 速度测试时开启了隔帧检测策略，默认检测间隔为 5 帧。
- 环境配置:
  - torch >= 1.7.1
  - onnxruntime 1.12.1
  - TensorRT 8.4.3.1
  - cuDNN 8.3.2
  - CUDA 11.3

| Detection Config                                                    | Pose Config                                                                                                     | Input Size<sup><br>(Det/Pose) | Model AP<sup><br>(COCO) | Pipeline AP<sup><br>(COCO) | Params (M)<sup><br>(Det/Pose) | Flops (G)<sup><br>(Det/Pose) | ncnn-Latency(ms)<sup><br>(Snapdragon 865) | ORT-Latency(ms)<sup><br>(i7-11700) | TRT-FP16-Latency(ms)<sup><br>(GTX 1660Ti) |                                                                                                                                 Download                                                                                                                                 |
| :------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------- | :---------------------------: | :---------------------: | :------------------------: | :---------------------------: | :--------------------------: | :---------------------------------------: | :--------------------------------: | :---------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [RTMDet-nano](./rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py) | [RTMPose-t](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-tiny_8xb256-420e_aic-coco-256x192.py) |      320x320<br>256x192       |      40.3<br>67.1       |            64.4            |         0.99<br/>3.34         |        0.31<br/>0.36         |                  18.780                   |               12.403               |                   2.467                   | [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth) |
| [RTMDet-nano](./rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py) | [RTMPose-s](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py)    |      320x320<br>256x192       |      40.3<br>71.1       |            68.5            |         0.99<br/>5.47         |        0.31<br/>0.68         |                  21.683                   |               16.658               |                   2.730                   |  [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth)   |
| [RTMDet-nano](./rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py) | [RTMPose-m](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-m_8xb256-420e_aic-coco-256x192.py)    |      320x320<br>256x192       |      40.3<br>75.3       |            73.2            |        0.99<br/>13.59         |        0.31<br/>1.93         |                  32.122                   |               26.613               |                   4.312                   |  [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth)   |
| [RTMDet-nano](./rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py) | [RTMPose-l](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py)    |      320x320<br>256x192       |      40.3<br>76.3       |            74.2            |        0.99<br/>27.66         |        0.31<br/>4.16         |                  47.642                   |               36.311               |                   4.644                   |  [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth)   |
| [RTMDet-m](./rtmdet/person/rtmdet_m_640-8xb32_coco-person.py)       | [RTMPose-m](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-m_8xb256-420e_aic-coco-256x192.py)    |      640x640<br>256x192       |      62.5<br>75.3       |            75.7            |        24.66<br/>13.59        |        38.95<br/>1.93        |                     -                     |                 -                  |                   6.923                   |    [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth)    |
| [RTMDet-m](./rtmdet/person/rtmdet_m_640-8xb32_coco-person.py)       | [RTMPose-l](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py)    |      640x640<br>256x192       |      62.5<br>76.3       |            76.6            |        24.66<br/>27.66        |        38.95<br/>4.16        |                     -                     |                 -                  |                   7.204                   |    [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth)    |

## 模型库

**说明**

- 此处提供的模型采用了多数据集联合训练以提高性能，模型指标不适用于学术比较。
- RTMPose 在更多公开数据集上的性能指标可以前往 [Model Zoo](https://mmpose.readthedocs.io/en/1.x/model_zoo_papers/algorithms.html) 查看。
- 表格中为开启了 Flip Test 的测试结果。
- 如果你有希望我们支持的数据集，欢迎[联系我们](https://uua478.fanqier.cn/f/xxmynrki)/[Google Questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSfzwWr3eNlDzhU98qzk2Eph44Zio6hi5r0iSwfO9wSARkHdWg/viewform?usp=sf_link)！

### 人体 2d 关键点 (17 Keypoints)

|   Config    | Input Size | AP<sup><br>(COCO) | Params(M) | FLOPS(G) | ORT-Latency(ms)<sup><br>(i7-11700) | TRT-FP16-Latency(ms)<sup><br>(GTX 1660Ti) | ncnn-FP16-Latency(ms)<sup><br>(Snapdragon 865) |    Logs    |    Download    |
| :---------: | :--------: | :---------------: | :-------: | :------: | :--------------------------------: | :---------------------------------------: | :--------------------------------------------: | :--------: | :------------: |
| [RTMPose-t](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-tiny_8xb256-420e_aic-coco-256x192.py) |  256x192   |       68.5        |   3.34    |   0.36   |                3.20                |                   1.06                    |                      9.02                      | [Log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.json) | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth) |
| [RTMPose-s](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py) |  256x192   |       72.2        |   5.47    |   0.68   |                4.48                |                   1.39                    |                     13.89                      | [Log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.json) | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth) |
| [RTMPose-m](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-m_8xb256-420e_aic-coco-256x192.py) |  256x192   |       75.8        |   13.59   |   1.93   |               11.06                |                   2.29                    |                     26.44                      | [Log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.json) | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth) |
| [RTMPose-l](./rtmpose/body_2d_keypoint/combined_datasets/aic-coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py) |  256x192   |       76.5        |   27.66   |   4.16   |               18.85                |                   3.46                    |                     45.37                      | [Log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.json) | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth) |

#### 模型剪枝

**说明**

- 模型剪枝由 [MMRazor](https://github.com/open-mmlab/mmrazor) 提供

|   Config    | Input Size | AP<sup><br>(COCO) | Params(M) | FLOPS(G) | ORT-Latency(ms)<sup><br>(i7-11700) | TRT-FP16-Latency(ms)<sup><br>(GTX 1660Ti) | ncnn-FP16-Latency(ms)<sup><br>(Snapdragon 865) |    Logs     |  Download   |
| :---------: | :--------: | :---------------: | :-------: | :------: | :--------------------------------: | :---------------------------------------: | :--------------------------------------------: | :---------: | :---------: |
| pruning-s-t |  256x192   |       69.2        |   3.42    |   0.34   |                 -                  |                     -                     |                       -                        | Coming soon | Coming soon |

### 人体全身 2d 关键点 (133 Keypoints)

| Config                         | Input Size | Whole AP | Whole AR | FLOPS(G) | ORT-Latency(ms)<sup><br>(i7-11700) | TRT-FP16-Latency(ms)<sup><br>(GTX 1660Ti) |             Logs             |             Download              |
| :----------------------------- | :--------: | :------: | :------: | :------: | :--------------------------------: | :---------------------------------------: | :--------------------------: | :-------------------------------: |
| [RTMPose-m](./config/wholebody_2d_keypoint/rtmpose-m_finetune-aic-coco_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |   60.4   |   66.7   |   2.22   |               13.50                |                   4.00                    | [Log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.json) | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth) |
| [RTMPose-l](./config/wholebody_2d_keypoint/rtmpose-l_finetune-aic-coco_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |   63.2   |   69.4   |   4.52   |               23.41                |                   5.67                    | [Log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.json) | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth) |
| [RTMPose-l](./config/wholebody_2d_keypoint/rtmpose-l_finetune-aic-coco_8xb64-270e_coco-wholebody-384x288.py) |  384x288   |   67.0   |   72.3   |  10.07   |               44.58                |                   7.68                    | [Log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.json) | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth) |

### 动物 2d 关键点 (17 Keypoints)

|            Config             | Input Size | AP<sup><br>(AP10K) | FLOPS(G) | ORT-Latency(ms)<sup><br>(i7-11700) | TRT-FP16-Latency(ms)<sup><br>(GTX 1660Ti) |             Logs             |             Download             |
| :---------------------------: | :--------: | :----------------: | :------: | :--------------------------------: | :---------------------------------------: | :--------------------------: | :------------------------------: |
| [RTMPose-m](./rtmpose/animal_2d_keypoint/ap10k/rtmpose-m_finetune-aic-coco_8xb64-210e_ap10k-256x256.py) |  256x256   |        72.2        |   2.57   |               14.157               |                   2.404                   | [Log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.json) | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth) |

### 脸部 2d 关键点

Coming soon!

### 手部 2d 关键点

Coming soon!

### 预训练模型

我们提供了 UDP 预训练的 CSPNeXt 模型参数，训练配置请参考 [pretrain_cspnext_udp folder](./rtmpose/pretrain_cspnext_udp/)。

|    Model     | Input Size | Params(M) | Flops(G) | AP<sup><br>(GT) | AR<sup><br>(GT) |                                                            Download                                                             |
| :----------: | :--------: | :-------: | :------: | :-------------: | :-------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
| CSPNeXt-tiny |  256x192   |   6.03    |   1.43   |      65.5       |      68.9       | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth) |
|  CSPNeXt-s   |  256x192   |   8.58    |   1.78   |      70.0       |      73.3       |  [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth)   |
|  CSPNeXt-m   |  256x192   |   13.05   |   3.06   |      74.8       |      77.7       |  [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth)   |
|  CSPNeXt-l   |  256x192   |   32.44   |   5.33   |      77.2       |      79.9       |  [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth)   |

我们提供了 ImageNet 分类训练的 CSPNeXt 模型参数，更多细节请参考 [RTMDet](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/rtmdet/README.md#classification)。

|    Model     | Input Size | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                                              Download                                                               |
| :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------------------------------------------------------------------------------------------: |
| CSPNeXt-tiny |  224x224   |   2.73    |   0.34   |   69.44   |   89.45   |    [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth)     |
|  CSPNeXt-s   |  224x224   |   4.89    |   0.66   |   74.41   |   92.23   |      [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth)      |
|  CSPNeXt-m   |  224x224   |   13.05   |   1.93   |   79.27   |   94.79   | [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth) |
|  CSPNeXt-l   |  224x224   |   27.16   |   4.19   |   81.30   |   95.62   | [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth) |

## 可视化

![vis_simcc](https://user-images.githubusercontent.com/13503330/219270443-421d9b02-fcec-46de-90f2-ce769c67575a.png)

## 部署教程

本教程将展示如何通过 [MMDeploy-1.x](https://github.com/open-mmlab/mmdeploy/tree/1.x) 部署 RTMPose 项目。

### 安装

在开始部署之前，首先你需要确保正确安装了 MMPose, MMDetection, MMDeploy，相关安装教程如下：

- [安装 MMPose 与 MMDetection](https://mmpose.readthedocs.io/zh_CN/1.x/installation.html)
- [安装 MMDeploy](https://mmdeploy.readthedocs.io/zh_CN/1.x/04-supported-codebases/mmpose.html)

根据部署后端的不同，有的后端需要对自定义算子进行编译，请根据需求前往对应的文档确保环境搭建正确：

- [ONNX](https://mmdeploy.readthedocs.io/zh_CN/1.x/05-supported-backends/onnxruntime.html)
- [TensorRT](https://mmdeploy.readthedocs.io/zh_CN/1.x/05-supported-backends/tensorrt.html)
- [OpenVINO](https://mmdeploy.readthedocs.io/zh_CN/1.x/05-supported-backends/openvino.html)
- [更多](https://github.com/open-mmlab/mmdeploy/tree/1.x/docs/en/05-supported-backends)

### 模型转换

在完成安装之后，你就可以开始模型部署了。通过 MMDeploy 提供的 `tools/deploy.py` 可以方便地将 Pytorch 模型转换到不同的部署后端。

我们本节演示将 RTMDet 和 RTMPose 模型导出为 ONNX 和 TensorRT 格式，如果你希望了解更多内容请前往 [MMDeploy 文档](https://mmdeploy.readthedocs.io/zh_CN/1.x/02-how-to-run/convert_model.html)。

- ONNX 配置

  \- RTMDet：[`detection_onnxruntime_static.py`](https://github.com/open-mmlab/mmdeploy/blob/1.x/configs/mmdet/detection/detection_onnxruntime_static.py)

  \- RTMPose：[`pose-detection_simcc_onnxruntime_dynamic.py`](https://github.com/open-mmlab/mmdeploy/blob/1.x/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py)

- TensorRT 配置

  \- RTMDet：[`detection_tensorrt_static-320x320.py`](https://github.com/open-mmlab/mmdeploy/blob/1.x/configs/mmdet/detection/detection_tensorrt_static-320x320.py)

  \- RTMPose：[`pose-detection_simcc_tensorrt_dynamic-256x192.py`](https://github.com/open-mmlab/mmdeploy/blob/1.x/configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py)

如果你需要对部署配置进行修改，请参考 [MMDeploy config tutorial](https://mmdeploy.readthedocs.io/zh_CN/1.x/02-how-to-run/write_config.html).

本教程中使用的文件结构如下：

```Python
|----mmdeploy
|----mmdetection
|----mmpose
|----rtmdet_nano
|    |----rtmdet_nano_320-8xb32_coco-person.py
|    |----rtmdet_nano.pth
|----rtmpose_m
     |----rtmpose-m_8xb256-420e_coco-256x192.py
     |----rtmpose_m.pth
```

#### ONNX

运行如下命令：

```Python
# 前往 mmdeploy 目录
cd ${PATH_TO_MMDEPLOY}

# 转换 RTMDet
python tools/deploy.py \
    configs/mmdet/detection/detection_onnxrumtime_static.py \
    {RTMPOSE_PROJECT}/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    ../rtmdet_nano/rtmdet_nano.pth \
    demo/resources/human-pose.jpg \
    --work-dir mmdeploy_models/mmdet/ort \
    --device cpu \
    --show

# 转换 RTMPose
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    {RTMPOSE_PROJECT}/rtmpose/body_2d_keypoint/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../rtmpose_m/rtmpose_m.pth \
    demo/resources/human-pose.jpg \
    --work-dir mmdeploy_models/mmpose/ort \
    --device cpu \
    --show
```

默认导出模型文件为 `{work-dir}/end2end.onnx`

#### TensorRT

运行如下命令：

```Python
# 前往 mmdeploy 目录
cd ${PATH_TO_MMDEPLOY}

# 转换 RTMDet
python tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_static-320x320.py \
    {RTMPOSE_PROJECT}/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    ../rtmdet_nano/rtmdet_nano.pth \
    demo/resources/human-pose.jpg \
    --work-dir mmdeploy_models/mmdet/trt \
    --device cuda:0 \
    --show

# 转换 RTMPose
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py \
    {RTMPOSE_PROJECT}/rtmpose/body_2d_keypoint/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../rtmpose_m/rtmpose_m.pth \
    demo/resources/human-pose.jpg \
    --work-dir mmdeploy_models/mmpose/trt \
    --device cuda:0 \
    --show
```

默认导出模型文件为 `{work-dir}/end2end.engine`

如果模型顺利导出，你将会看到样例图片上的检测结果：

![convert_models](https://user-images.githubusercontent.com/13503330/217726963-7815dd01-561a-4605-b0c6-07b6fe1956c3.png)

#### 高级设置

如果需要使用 TensorRT-FP16，你可以通过修改以下配置开启：

```Python
# in MMDeploy config
backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True  # 打开 fp16
    ))
```

### SDK 推理

要进行 Pipeline 推理，需要先用 MMDeploy 导出 SDK 版本的 det 和 pose 模型，只需要在参数中加上`--dump-info`。

此处以 onnxruntime 的 cpu 模型为例，运行如下命令：

```Python
# RTMDet
python tools/deploy.py \
    configs/mmdet/detection/detection_onnxrumtime_dynamic.py \
    {RTMPOSE_PROJECT}/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    ../rtmdet_nano/rtmdet_nano.pth \
    demo/resources/human-pose.jpg \
    --work-dir mmdeploy_models/mmdet/sdk \
    --device cpu \
    --show \
    --dump-info  # 导出 sdk info

# RTMPose
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    {RTMPOSE_PROJECT}/rtmpose/body_2d_keypoint/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../rtmpose_m/rtmpose_m.pth \
    demo/resources/human-pose.jpg \
    --work-dir mmdeploy_models/mmpose/sdk \
    --device cpu \
    --show \
    --dump-info  # 导出 sdk info
```

默认会导出三个 json 文件：

```Python
|----sdk
     |----end2end.onnx    # ONNX model
     |----end2end.engine  # TensorRT engine file

     |----pipeline.json   #
     |----deploy.json     # json files for the SDK
     |----detail.json     #
```

#### Python API

```Python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import numpy as np
from mmdeploy_python import PoseDetector

def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('device_name', help='name of device, cuda or cpu')
    parser.add_argument(
        'model_path',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('image_path', help='path of an image')
    parser.add_argument(
        '--bbox',
        default=None,
        nargs='+',
        type=int,
        help='bounding box of an object in format (x, y, w, h)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    img = cv2.imread(args.image_path)

    detector = PoseDetector(
        model_path=args.model_path, device_name=args.device_name, device_id=0)

    if args.bbox is None:
        result = detector(img)
    else:
        # converter (x, y, w, h) -> (left, top, right, bottom)
        print(args.bbox)
        bbox = np.array(args.bbox, dtype=int)
        bbox[2:] += bbox[:2]
        result = detector(img, bbox)
    print(result)

    _, point_num, _ = result.shape
    points = result[:, :, :2].reshape(point_num, 2)
    for [x, y] in points.astype(int):
        cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

    cv2.imwrite('output_pose.png', img)

if __name__ == '__main__':
    main()
```

#### C++ API

```Python
#include "mmdeploy/detector.hpp"

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "utils/argparse.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Model path");
DEFINE_ARG_string(image, "Input image path");
DEFINE_string(device, "cpu", R"(Device name, e.g. "cpu", "cuda")");
DEFINE_string(output, "detector_output.jpg", "Output image path");

DEFINE_double(det_thr, .5, "Detection score threshold");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  cv::Mat img = cv::imread(ARGS_image);
  if (img.empty()) {
    fprintf(stderr, "failed to load image: %s\n", ARGS_image.c_str());
    return -1;
  }

  // construct a detector instance
  mmdeploy::Detector detector(mmdeploy::Model{ARGS_model}, mmdeploy::Device{FLAGS_device});

  // apply the detector, the result is an array-like class holding references to
  // `mmdeploy_detection_t`, will be released automatically on destruction
  mmdeploy::Detector::Result dets = detector.Apply(img);

  // visualize
  utils::Visualize v;
  auto sess = v.get_session(img);
  int count = 0;
  for (const mmdeploy_detection_t& det : dets) {
    if (det.score > FLAGS_det_thr) {  // filter bboxes
      sess.add_det(det.bbox, det.label_id, det.score, det.mask, count++);
    }
  }

  if (!FLAGS_output.empty()) {
    cv::imwrite(FLAGS_output, sess.get());
  }

  return 0;
}
```

对于 C++ API 示例，请将 MMDeploy 加入到 CMake 项目中：

```CMake
find_package(MMDeploy REQUIRED)
target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
```

#### 其他语言

- [C# API 示例](https://github.com/open-mmlab/mmdeploy/tree/1.x/demo/csharp)
- [JAVA API 示例](https://github.com/open-mmlab/mmdeploy/tree/1.x/demo/java)

### Pipeline 推理

#### 图片推理

如果用户有跟随 MMDeploy 安装教程进行正确编译，在 `mmdeploy/build/bin/` 路径下会看到 `det_pose` 的可执行文件。

```Python
# 前往 mmdeploy 目录
cd ${PATH_TO_MMDEPLOY}/build/bin/

# 单张图片推理
./det_pose {det work-dir} {pose work-dir} {your_img.jpg} --device cpu
```

**API** **示例**

\- [`det_pose.py`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/demo/python/det_pose.py)

\- [`det_pose.cxx`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/demo/csrc/cpp/det_pose.cxx)

#### 视频推理

如果用户有跟随 MMDeploy 安装教程进行正确编译，在 `mmdeploy/build/bin/` 路径下会看到 `pose_tracker` 的可执行文件。

```Python
# 前往 mmdeploy 目录
cd ${PATH_TO_MMDEPLOY}/build/bin/

# 视频推理
./pose_tracker {det work-dir} {pose work-dir} {your_video.mp4} --device cpu
```

**API** **示例**

\- [`pose_tracker.py`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/demo/python/pose_tracker.py)

\- [`pose_tracker.cxx`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/demo/csrc/cpp/pose_tracker.cxx)

### 常用功能

#### 模型测速

如果需要测试模型在部署框架下的推理速度，MMDeploy 提供了方便的 `tools/profiler.py` 脚本。

此处以 onnxruntime 的 cpu 模型为例。进入 mmdeploy 目录，首先按照 Step 2 中的步骤导出 ONNX 模型。用户需要准备一个存放测试图片的文件夹`./test_images`，profiler 将随机从该目录下抽取图片用于模型测速。

```Python
python tools/profiler.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    {RTMPOSE_PROJECT}/rtmpose/body_2d_keypoint/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../test_images \
    --model {WORK_DIR}/end2end.onnx \
    --shape 256x192 \
    --device cpu \
    --warmup 50 \
    --num-iter 200
```

测试结果如下：

```Python
01/30 15:06:35 - mmengine - INFO - [onnxruntime]-70 times per count: 8.73 ms, 114.50 FPS
01/30 15:06:36 - mmengine - INFO - [onnxruntime]-90 times per count: 9.05 ms, 110.48 FPS
01/30 15:06:37 - mmengine - INFO - [onnxruntime]-110 times per count: 9.87 ms, 101.32 FPS
01/30 15:06:37 - mmengine - INFO - [onnxruntime]-130 times per count: 9.99 ms, 100.10 FPS
01/30 15:06:38 - mmengine - INFO - [onnxruntime]-150 times per count: 10.39 ms, 96.29 FPS
01/30 15:06:39 - mmengine - INFO - [onnxruntime]-170 times per count: 10.77 ms, 92.86 FPS
01/30 15:06:40 - mmengine - INFO - [onnxruntime]-190 times per count: 10.98 ms, 91.05 FPS
01/30 15:06:40 - mmengine - INFO - [onnxruntime]-210 times per count: 11.19 ms, 89.33 FPS
01/30 15:06:41 - mmengine - INFO - [onnxruntime]-230 times per count: 11.16 ms, 89.58 FPS
01/30 15:06:42 - mmengine - INFO - [onnxruntime]-250 times per count: 11.06 ms, 90.41 FPS
----- Settings:
+------------+---------+
| batch size |    1    |
|   shape    | 256x192 |
| iterations |   200   |
|   warmup   |    50   |
+------------+---------+
----- Results:
+--------+------------+---------+
| Stats  | Latency/ms |   FPS   |
+--------+------------+---------+
|  Mean  |   11.060   |  90.412 |
| Median |   11.852   |  84.375 |
|  Min   |   7.812    | 128.007 |
|  Max   |   13.690   |  73.044 |
+--------+------------+---------+
```

如果你希望详细了解 profiler 的更多参数设置与功能，可以前往 [Profiler Docs](https://mmdeploy.readthedocs.io/en/1.x/02-how-to-run/useful_tools.html#profiler)

#### 精度验证

如果需要测试模型在部署框架下的推理精度，MMDeploy 提供了方便的 `tools/test.py` 脚本。

```CMake
python tools/test.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    {RTMPOSE_PROJECT}/rtmpose/body_2d_keypoint/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
    --model {PATH_TO_MODEL}/rtmpose_m.pth \
    --device cpu
```

详细内容请参考 [MMDeploys Docs](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/docs/zh_cn/02-how-to-run/profile_model.md)
