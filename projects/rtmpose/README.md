<div align="center">
  <img width="100%" src="https://github.com/open-mmlab/mmpose/assets/13503330/5b637d76-41dd-4376-9a7f-854cd120799d"/>
</div>

# RTMPose: Real-Time Multi-Person Pose Estimation toolkit based on MMPose

> [RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose](https://arxiv.org/abs/2303.07399)

<div align="center">

English | [简体中文](README_CN.md)

</div>

______________________________________________________________________

## Abstract

Recent studies on 2D pose estimation have achieved excellent performance on public benchmarks, yet its application in the industrial community still suffers from heavy model parameters and high latency.
In order to bridge this gap, we empirically study five aspects that affect the performance of multi-person pose estimation algorithms: paradigm, backbone network, localization algorithm, training strategy, and deployment inference, and present a high-performance real-time multi-person pose estimation framework, **RTMPose**, based on MMPose.
Our RTMPose-m achieves **75.8% AP** on COCO with **90+ FPS** on an Intel i7-11700 CPU and **430+ FPS** on an NVIDIA GTX 1660 Ti GPU.
To further evaluate RTMPose's capability in critical real-time applications, we also report the performance after deploying on the mobile device. Our RTMPose-s achieves **72.2% AP** on COCO with **70+ FPS** on a Snapdragon 865 chip, outperforming existing open-source libraries.
With the help of MMDeploy, our project supports various platforms like CPU, GPU, NVIDIA Jetson, and mobile devices and multiple inference backends such as ONNXRuntime, TensorRT, ncnn, etc.

![rtmpose_intro](https://user-images.githubusercontent.com/13503330/219269619-935499e5-bdd9-49ea-8104-3c7796dbd862.png)

______________________________________________________________________

## 📄 Table of Contents

- [🥳 🚀 What's New](#--whats-new-)
- [📖 Introduction](#-introduction-)
- [🙌 Community](#-community-)
- [⚡ Pipeline Performance](#-pipeline-performance-)
- [📊 Model Zoo](#-model-zoo-)
- [👀 Visualization](#-visualization-)
- [😎 Get Started](#-get-started-)
- [👨‍🏫 How to Train](#-how-to-train-)
- [🏗️ How to Deploy](#️-how-to-deploy-)
- [📚 Common Usage](#️-common-usage-)
  - [🚀 Inference Speed Test](#-inference-speed-test-)
  - [📊 Model Test](#-model-test-)
- [📜 Citation](#-citation-)

## 🥳 🚀 What's New [🔝](#-table-of-contents)

- Jun. 2023:
  - Release 26-keypoint Body models trained on combined datasets.
- May. 2023:
  - Add [code examples](./examples/) of RTMPose.
  - Release Hand, Face, Body models trained on combined datasets.
- Mar. 2023: RTMPose is released. RTMPose-m runs at 430+ FPS and achieves 75.8 mAP on COCO val set.

## 📖 Introduction [🔝](#-table-of-contents)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/221138554-110240d8-e887-4b9a-90b1-2fbdc982e9de.gif" width=400 height=300/><img src="https://user-images.githubusercontent.com/13503330/221125176-85015a13-9648-4f0d-a17c-1cbb469efacf.gif" width=250 height=300/><img src="https://user-images.githubusercontent.com/13503330/221125310-7eeb2212-907e-427f-97af-af799d70a4c5.gif" width=250 height=300/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/mmpose/assets/13503330/38aa345e-4ceb-4e73-bc37-5e082735e336" width=450 height=300/><img src="https://user-images.githubusercontent.com/13503330/221125888-15c20faf-0ad5-4afb-828b-a71ccb064582.gif" width=450 height=300/>
</div>
<div align=center>
<img src="https://github.com/open-mmlab/mmpose/assets/13503330/2ecbf9f4-6963-4a14-9801-da10c0a65dac" width=300 height=350/><img src="https://user-images.githubusercontent.com/13503330/221138017-10431ab4-e515-4c32-8fa7-8748e2d17a58.gif" width=600 height=350/>
</div>

### ✨ Major Features

- 🚀 **High efficiency and high accuracy**

  | Model | AP(COCO) | CPU-FPS | GPU-FPS |
  | :---: | :------: | :-----: | :-----: |
  |   t   |   68.5   |  300+   |  940+   |
  |   s   |   72.2   |  200+   |  710+   |
  |   m   |   75.8   |   90+   |  430+   |
  |   l   |   76.5   |   50+   |  280+   |

- 🛠️ **Easy to deploy**

  - Step-by-step deployment tutorials.
  - Support various backends including
    - ONNX
    - TensorRT
    - ncnn
    - OpenVINO
    - etc.
  - Support various platforms including
    - Linux
    - Windows
    - NVIDIA Jetson
    - ARM
    - etc.

- 🏗️ **Design for practical applications**

  - Pipeline inference API and SDK for
    - Python
    - C++
    - C#
    - JAVA
    - etc.

## 🙌 Community [🔝](#-table-of-contents)

RTMPose is a long-term project dedicated to the training, optimization and deployment of high-performance real-time pose estimation algorithms in practical scenarios, so we are looking forward to the power from the community. Welcome to share the training configurations and tricks based on RTMPose in different business applications to help more community users!

✨ ✨ ✨

- **If you are a new user of RTMPose, we eagerly hope you can fill out this [Google Questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSfzwWr3eNlDzhU98qzk2Eph44Zio6hi5r0iSwfO9wSARkHdWg/viewform?usp=sf_link)/[Chinese version](https://uua478.fanqier.cn/f/xxmynrki), it's very important for our work!**

✨ ✨ ✨

Feel free to join our community group for more help:

- WeChat Group:

<div align=left>
<img src="https://user-images.githubusercontent.com/13503330/222647056-875bed70-85ec-455c-9016-c024772915c4.jpg" width=200 />
</div>

- Discord Group:
  - 🙌 https://discord.gg/raweFPmdzG 🙌

## ⚡ Pipeline Performance [🔝](#-table-of-contents)

**Notes**

- Pipeline latency is tested under skip-frame settings, the detection interval is 5 frames by defaults.
- Flip test is NOT used.
- Env Setup:
  - torch >= 1.7.1
  - onnxruntime 1.12.1
  - TensorRT 8.4.3.1
  - ncnn 20221128
  - cuDNN 8.3.2
  - CUDA 11.3

| Detection Config                                                    | Pose Config                                                                   | Input Size<sup><br>(Det/Pose) | Model AP<sup><br>(COCO) | Pipeline AP<sup><br>(COCO) | Params (M)<sup><br>(Det/Pose) | Flops (G)<sup><br>(Det/Pose) | ORT-Latency(ms)<sup><br>(i7-11700) | TRT-FP16-Latency(ms)<sup><br>(GTX 1660Ti) |                                                                                                                                  Download                                                                                                                                  |
| :------------------------------------------------------------------ | :---------------------------------------------------------------------------- | :---------------------------: | :---------------------: | :------------------------: | :---------------------------: | :--------------------------: | :--------------------------------: | :---------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [RTMDet-nano](./rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py) | [RTMPose-t](./rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py) |      320x320<br>256x192       |      40.3<br>67.1       |            64.4            |         0.99<br/>3.34         |        0.31<br/>0.36         |               12.403               |                   2.467                   | [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth) |
| [RTMDet-nano](./rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py) | [RTMPose-s](./rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py) |      320x320<br>256x192       |      40.3<br>71.1       |            68.5            |         0.99<br/>5.47         |        0.31<br/>0.68         |               16.658               |                   2.730                   |  [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth)   |
| [RTMDet-nano](./rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py) | [RTMPose-m](./rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py) |      320x320<br>256x192       |      40.3<br>75.3       |            73.2            |        0.99<br/>13.59         |        0.31<br/>1.93         |               26.613               |                   4.312                   |  [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth)   |
| [RTMDet-nano](./rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py) | [RTMPose-l](./rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py) |      320x320<br>256x192       |      40.3<br>76.3       |            74.2            |        0.99<br/>27.66         |        0.31<br/>4.16         |               36.311               |                   4.644                   |  [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth)   |
| [RTMDet-m](./rtmdet/person/rtmdet_m_640-8xb32_coco-person.py)       | [RTMPose-m](./rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py) |      640x640<br>256x192       |      62.5<br>75.3       |            75.7            |        24.66<br/>13.59        |        38.95<br/>1.93        |                 -                  |                   6.923                   |    [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth)    |
| [RTMDet-m](./rtmdet/person/rtmdet_m_640-8xb32_coco-person.py)       | [RTMPose-l](./rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py) |      640x640<br>256x192       |      62.5<br>76.3       |            76.6            |        24.66<br/>27.66        |        38.95<br/>4.16        |                 -                  |                   7.204                   |    [det](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth)<br/>[pose](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth)    |

## 📊 Model Zoo [🔝](#-table-of-contents)

**Notes**

- Since all models are trained on multi-domain combined datasets for practical applications, results are **not** suitable for academic comparison.
- More results of RTMPose on public benchmarks can refer to [Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html)
- Flip test is used.
- Inference speed measured on more hardware platforms can refer to [Benchmark](./benchmark/README.md)
- If you have datasets you would like us to support, feel free to [contact us](https://docs.google.com/forms/d/e/1FAIpQLSfzwWr3eNlDzhU98qzk2Eph44Zio6hi5r0iSwfO9wSARkHdWg/viewform?usp=sf_link)/[联系我们](https://uua478.fanqier.cn/f/xxmynrki).

### Body 2d

#### 17 Keypoints

- Keypoints are defined as [COCO](http://cocodataset.org/). For details please refer to the [meta info](/configs/_base_/datasets/coco.py).
- <img src="https://github.com/open-mmlab/mmpose/assets/13503330/2417e4f7-2203-468f-bad0-e7a6a6bf8251" height="300px">

<details close>
<summary><b>AIC+COCO</b></summary>

|                                    Config                                     | Input Size | AP<sup><br>(COCO) | PCK@0.1<sup><br>(Body8) | AUC<sup><br>(Body8) | Params<sup><br>(M) | FLOPS<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) | ncnn-FP16-Latency<sup><br>(ms)<sup><br>(Snapdragon 865) |                                                                    Download                                                                     |
| :---------------------------------------------------------------------------: | :--------: | :---------------: | :---------------------: | :-----------------: | :----------------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :-----------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
| [RTMPose-t](./rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py) |  256x192   |       68.5        |          91.28          |        63.38        |        3.34        |       0.36        |                    3.20                     |                        1.06                        |                          9.02                           | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth) |
| [RTMPose-s](./rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   |       72.2        |          92.95          |        66.19        |        5.47        |       0.68        |                    4.48                     |                        1.39                        |                          13.89                          |  [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth)   |
| [RTMPose-m](./rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   |       75.8        |          94.13          |        68.53        |       13.59        |       1.93        |                    11.06                    |                        2.29                        |                          26.44                          |  [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth)   |
| [RTMPose-l](./rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   |       76.5        |          94.35          |        68.98        |       27.66        |       4.16        |                    18.85                    |                        3.46                        |                          45.37                          |  [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth)   |
| [RTMPose-m](./rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-384x288.py) |  384x288   |       77.0        |          94.32          |        69.85        |       13.72        |       4.33        |                    24.78                    |                        3.66                        |                            -                            |  [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-384x288-a62a0b32_20230228.pth)   |
| [RTMPose-l](./rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py) |  384x288   |       77.3        |          94.54          |        70.14        |       27.79        |       9.35        |                      -                      |                        6.05                        |                            -                            |  [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth)   |

</details>

<details open>
<summary><b>Body8</b></summary>

- `*` denotes model trained on 7 public datasets:
  - [AI Challenger](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#aic)
  - [MS COCO](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#coco)
  - [CrowdPose](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#crowdpose)
  - [MPII](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#mpii)
  - [sub-JHMDB](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#sub-jhmdb-dataset)
  - [Halpe](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe)
  - [PoseTrack18](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#posetrack18)
- `Body8` denotes the addition of the [OCHuman](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#ochuman) dataset, in addition to the 7 datasets mentioned above, for evaluation.

|                                     Config                                      | Input Size | AP<sup><br>(COCO) | PCK@0.1<sup><br>(Body8) | AUC<sup><br>(Body8) | Params<sup><br>(M) | FLOPS<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) | ncnn-FP16-Latency<sup><br>(ms)<sup><br>(Snapdragon 865) |                                                                Download                                                                |
| :-----------------------------------------------------------------------------: | :--------: | :---------------: | :---------------------: | :-----------------: | :----------------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :-----------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------: |
| [RTMPose-t\*](./rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py) |  256x192   |       65.9        |          91.44          |        63.18        |        3.34        |       0.36        |                    3.20                     |                        1.06                        |                          9.02                           | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth) |
| [RTMPose-s\*](./rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   |       69.7        |          92.45          |        65.15        |        5.47        |       0.68        |                    4.48                     |                        1.39                        |                          13.89                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth) |
| [RTMPose-m\*](./rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   |       74.9        |          94.25          |        68.59        |       13.59        |       1.93        |                    11.06                    |                        2.29                        |                          26.44                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth) |
| [RTMPose-l\*](./rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   |       76.7        |          95.08          |        70.14        |       27.66        |       4.16        |                    18.85                    |                        3.46                        |                          45.37                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth) |
| [RTMPose-m\*](./rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-384x288.py) |  384x288   |       76.6        |          94.64          |        70.38        |       13.72        |       4.33        |                    24.78                    |                        3.66                        |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth) |
| [RTMPose-l\*](./rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py) |  384x288   |       78.3        |          95.36          |        71.58        |       27.79        |       9.35        |                      -                      |                        6.05                        |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth) |
| [RTMPose-x\*](./rtmpose/body_2d_keypoint/rtmpose-x_8xb256-700e_coco-384x288.py) |  384x288   |       78.8        |            -            |          -          |       49.43        |       17.22       |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth) |

</details>

#### 26 Keypoints

- Keypoints are defined as [Halpe26](https://github.com/Fang-Haoshu/Halpe-FullBody/). For details please refer to the [meta info](/configs/_base_/datasets/halpe26.py).
- <img src="https://github.com/open-mmlab/mmpose/assets/13503330/f28ab3ba-833d-4ca7-8739-f97e6cafbab7" height="300px">
- Models are trained and evaluated on `Body8`.

|                                          Config                                           | Input Size | PCK@0.1<sup><br>(Body8) | AUC<sup><br>(Body8) | Params(M) | FLOPS(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) | ncnn-FP16-Latency<sup><br>(ms)<sup><br>(Snapdragon 865) |                                                                    Download                                                                    |
| :---------------------------------------------------------------------------------------: | :--------: | :---------------------: | :-----------------: | :-------: | :------: | :-----------------------------------------: | :------------------------------------------------: | :-----------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [RTMPose-t\*](./rtmpose/body_2d_keypoint/rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py) |  256x192   |          91.89          |        66.35        |   3.51    |   0.37   |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth) |
| [RTMPose-s\*](./rtmpose/body_2d_keypoint/rtmpose-s_8xb1024-700e_body8-halpe26-256x192.py) |  256x192   |          93.01          |        68.62        |   5.70    |   0.70   |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth) |
| [RTMPose-m\*](./rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py)  |  256x192   |          94.75          |        71.91        |   13.93   |   1.95   |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth) |
| [RTMPose-l\*](./rtmpose/body_2d_keypoint/rtmpose-l_8xb512-700e_body8-halpe26-256x192.py)  |  256x192   |          95.37          |        73.19        |   28.11   |   4.19   |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth) |
| [RTMPose-m\*](./rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py)  |  384x288   |          95.15          |        73.56        |   14.06   |   4.37   |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth) |
| [RTMPose-l\*](./rtmpose/body_2d_keypoint/rtmpose-l_8xb512-700e_body8-halpe26-384x288.py)  |  384x288   |          95.56          |        74.38        |   28.24   |   9.40   |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.pth) |
| [RTMPose-x\*](./rtmpose/body_2d_keypoint/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py)  |  384x288   |          95.74          |        74.82        |   50.00   |  17.29   |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth) |

#### Model Pruning

**Notes**

- Model pruning is supported by [MMRazor](https://github.com/open-mmlab/mmrazor)

|          Config           | Input Size | AP<sup><br>(COCO) | Params<sup><br>(M) | FLOPS<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) | ncnn-FP16-Latency<sup><br>(ms)<sup><br>(Snapdragon 865) |                                                                      Download                                                                      |
| :-----------------------: | :--------: | :---------------: | :----------------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :-----------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMPose-s-aic-coco-pruned |  256x192   |       69.4        |        3.43        |       0.35        |                      -                      |                         -                          |                            -                            | [Model](https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/rtmpose-s/group_fisher_finetune_rtmpose-s_8xb256-420e_aic-coco-256x192.pth) |

For more details, please refer to [GroupFisher Pruning for RTMPose](./rtmpose/pruning/README.md).

### WholeBody 2d (133 Keypoints)

- Keypoints are defined as [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/). For details please refer to the [meta info](/configs/_base_/datasets/coco_wholebody.py).
- <img src="https://user-images.githubusercontent.com/100993824/227770977-c8f00355-c43a-467e-8444-d307789cf4b2.png" height="300px">

| Config                          | Input Size | Whole AP | Whole AR | FLOPS<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) |             Download              |
| :------------------------------ | :--------: | :------: | :------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :-------------------------------: |
| [RTMPose-m](./rtmpose/wholebody_2d_keypoint/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |   58.2   |   67.4   |       2.22        |                    13.50                    |                        4.00                        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth) |
| [RTMPose-l](./rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |   61.1   |   70.0   |       4.52        |                    23.41                    |                        5.67                        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth) |
| [RTMPose-l](./rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py) |  384x288   |   64.8   |   73.0   |       10.07       |                    44.58                    |                        7.68                        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth) |
| [RTMPose-x](./rtmpose/wholebody_2d_keypoint/rtmpose-x_8xb32-270e_coco-wholebody-384x288.py) |  384x288   |   65.3   |   73.3   |       18.1        |                      -                      |                         -                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth) |

### Animal 2d (17 Keypoints)

- Keypoints are defined as [AP-10K](https://github.com/AlexTheBad/AP-10K/). For details please refer to the [meta info](/configs/_base_/datasets/ap10k.py).
- <img src="https://user-images.githubusercontent.com/100993824/227797151-091dc21a-d944-49c9-8b62-cc47fa89e69f.png" height="300px">

|             Config             | Input Size | AP<sup><br>(AP10K) | FLOPS<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) |             Download             |
| :----------------------------: | :--------: | :----------------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :------------------------------: |
| [RTMPose-m](./rtmpose/animal_2d_keypoint/rtmpose-m_8xb64-210e_ap10k-256x256.py) |  256x256   |        72.2        |       2.57        |                   14.157                    |                       2.404                        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth) |

### Face 2d (106 Keypoints)

- Keypoints are defined as [LaPa](https://github.com/JDAI-CV/lapa-dataset). For details please refer to the [meta info](/configs/_base_/datasets/lapa.py).
- <img src="https://github.com/open-mmlab/mmpose/assets/13503330/30fa583e-500c-4356-ac5a-7e9d7d18381a" height="300px">

<details open>
<summary><b>Face6</b></summary>

- `Face6` and `*` denote model trained on 6 public datasets:
  - [COCO-Wholebody-Face](https://github.com/jin-s13/COCO-WholeBody/)
  - [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)
  - [300W](https://ibug.doc.ic.ac.uk/resources/300-W/)
  - [COFW](http://www.vision.caltech.edu/xpburgos/ICCV13/)
  - [Halpe](https://github.com/Fang-Haoshu/Halpe-FullBody/)
  - [LaPa](https://github.com/JDAI-CV/lapa-dataset)

|             Config             | Input Size | NME<sup><br>(LaPa) | FLOPS<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) |             Download             |
| :----------------------------: | :--------: | :----------------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :------------------------------: |
| [RTMPose-t\*](./rtmpose/face_2d_keypoint/rtmpose-t_8xb256-120e_lapa-256x256.py) |  256x256   |        1.67        |       0.652       |                      -                      |                         -                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-face6_pt-in1k_120e-256x256-df79d9a5_20230529.pth) |
| [RTMPose-s\*](./rtmpose/face_2d_keypoint/rtmpose-s_8xb256-120e_lapa-256x256.py) |  256x256   |        1.59        |       1.119       |                      -                      |                         -                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-face6_pt-in1k_120e-256x256-d779fdef_20230529.pth) |
| [RTMPose-m\*](./rtmpose/face_2d_keypoint/rtmpose-m_8xb256-120e_lapa-256x256.py) |  256x256   |        1.44        |       2.852       |                      -                      |                         -                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth) |

</details>

### Hand 2d (21 Keypoints)

- Keypoints are defined as [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/). For details please refer to the [meta info](/configs/_base_/datasets/coco_wholebody_hand.py).
- <img src="https://user-images.githubusercontent.com/100993824/227771101-03a27bd8-ccc0-4eb9-a111-660f191a7a16.png" height="300px">

|       Detection Config        | Input Size | Model AP<sup><br>(OneHand10K) | Flops<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) |        Download        |
| :---------------------------: | :--------: | :---------------------------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :--------------------: |
| [RTMDet-nano<sup><br>(alpha version)](./rtmdet/hand/rtmdet_nano_320-8xb32_hand.py) |  320x320   |             76.0              |       0.31        |                      -                      |                         -                          | [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth) |

<details open>
<summary><b>Hand5</b></summary>

- `Hand5` and `*` denote model trained on 5 public datasets:
  - [COCO-Wholebody-Hand](https://github.com/jin-s13/COCO-WholeBody/)
  - [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)
  - [FreiHand2d](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
  - [RHD2d](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)
  - [Halpe](https://github.com/Fang-Haoshu/Halpe-FullBody/)

|                                                        Config                                                         | Input Size | PCK@0.2<sup><br>(COCO-Wholebody-Hand) | PCK@0.2<sup><br>(Hand5) | AUC<sup><br>(Hand5) | FLOPS<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) |                                                                 Download                                                                 |
| :-------------------------------------------------------------------------------------------------------------------: | :--------: | :-----------------------------------: | :---------------------: | :-----------------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| [RTMPose-m\*<sup><br>(alpha version)](./rtmpose/hand_2d_keypoint/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |                 81.5                  |          96.4           |        83.9         |       2.581       |                      -                      |                         -                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth) |

</details>

### Pretrained Models

We provide the UDP pretraining configs of the CSPNeXt backbone. Find more details in the [pretrain_cspnext_udp folder](./rtmpose/pretrain_cspnext_udp/).

<details close>
<summary><b>AIC+COCO</b></summary>

|    Model     | Input Size | Params<sup><br>(M) | Flops<sup><br>(G) | AP<sup><br>(GT) | AR<sup><br>(GT) |                                                     Download                                                      |
| :----------: | :--------: | :----------------: | :---------------: | :-------------: | :-------------: | :---------------------------------------------------------------------------------------------------------------: |
| CSPNeXt-tiny |  256x192   |        6.03        |       1.43        |      65.5       |      68.9       | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth) |
|  CSPNeXt-s   |  256x192   |        8.58        |       1.78        |      70.0       |      73.3       | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth) |
|  CSPNeXt-m   |  256x192   |       17.53        |       3.05        |      74.8       |      77.7       | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth) |
|  CSPNeXt-l   |  256x192   |       32.44        |       5.32        |      77.2       |      79.9       | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth) |

</details>

<details open>
<summary><b>Body8</b></summary>

- `*` denotes model trained on 7 public datasets:
  - [AI Challenger](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#aic)
  - [MS COCO](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#coco)
  - [CrowdPose](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#crowdpose)
  - [MPII](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#mpii)
  - [sub-JHMDB](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#sub-jhmdb-dataset)
  - [Halpe](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe)
  - [PoseTrack18](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#posetrack18)
- `Body8` denotes the addition of the [OCHuman](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#ochuman) dataset, in addition to the 7 datasets mentioned above, for evaluation.

|     Model      | Input Size | Params<sup><br>(M) | Flops<sup><br>(G) | AP<sup><br>(COCO) | PCK@0.2<sup><br>(Body8) | AUC<sup><br>(Body8) |                                      Download                                      |
| :------------: | :--------: | :----------------: | :---------------: | :---------------: | :---------------------: | :-----------------: | :--------------------------------------------------------------------------------: |
| CSPNeXt-tiny\* |  256x192   |        6.03        |       1.43        |       65.9        |          96.34          |        63.80        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-tiny_udp-body7_210e-256x192-a3775292_20230504.pth) |
|  CSPNeXt-s\*   |  256x192   |        8.58        |       1.78        |       68.7        |          96.59          |        64.92        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-body7_210e-256x192-8c9ccbdb_20230504.pth) |
|  CSPNeXt-m\*   |  256x192   |       17.53        |       3.05        |       73.7        |          97.42          |        68.19        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-body7_210e-256x192-e0c9327b_20230504.pth) |
|  CSPNeXt-l\*   |  256x192   |       32.44        |       5.32        |       75.7        |          97.76          |        69.57        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-body7_210e-256x192-5e9558ef_20230504.pth) |
|  CSPNeXt-m\*   |  384x288   |       17.53        |       6.86        |       75.8        |          97.60          |        70.18        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-body7_210e-384x288-b9bc2b57_20230504.pth) |
|  CSPNeXt-l\*   |  384x288   |       32.44        |       11.96       |       77.2        |          97.89          |        71.23        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-body7_210e-384x288-b15bc30d_20230504.pth) |
|  CSPNeXt-x\*   |  384x288   |       54.92        |       19.96       |       78.1        |          98.00          |        71.79        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-x_udp-body7_210e-384x288-d28b58e6_20230529.pth) |

</details>

#### ImageNet

We also provide the ImageNet classification pre-trained weights of the CSPNeXt backbone. Find more details in [RTMDet](https://github.com/open-mmlab/mmdetection/blob/latest/configs/rtmdet/README.md#classification).

|    Model     | Input Size | Params<sup><br>(M) | Flops<sup><br>(G) | Top-1 (%) | Top-5 (%) |                                                           Download                                                            |
| :----------: | :--------: | :----------------: | :---------------: | :-------: | :-------: | :---------------------------------------------------------------------------------------------------------------------------: |
| CSPNeXt-tiny |  224x224   |        2.73        |       0.34        |   69.44   |   89.45   | [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth)  |
|  CSPNeXt-s   |  224x224   |        4.89        |       0.66        |   74.41   |   92.23   |   [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth)   |
|  CSPNeXt-m   |  224x224   |       13.05        |       1.93        |   79.27   |   94.79   | [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth) |
|  CSPNeXt-l   |  224x224   |       27.16        |       4.19        |   81.30   |   95.62   | [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth) |
|  CSPNeXt-x   |  224x224   |       48.85        |       7.76        |   82.10   |   95.69   | [Model](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-x_8xb256-rsb-a1-600e_in1k-b3f78edd.pth) |

## 👀 Visualization [🔝](#-table-of-contents)

<div align=center>
<img src='https://user-images.githubusercontent.com/13503330/221795678-2c4ae2ec-ac23-4368-8083-0ebeb29f0d3c.gif' width=900/>
<img src="https://user-images.githubusercontent.com/13503330/219270443-421d9b02-fcec-46de-90f2-ce769c67575a.png" width=900 />
</div>

## 😎 Get Started [🔝](#-table-of-contents)

We provide two appoaches to try RTMPose:

- MMPose demo scripts
- Pre-compiled MMDeploy SDK (Recommend, 6-10 times faster)

### MMPose demo scripts

MMPose provides demo scripts to conduct [inference with existing models](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html).

**Note:**

- Inferencing with Pytorch can not reach the maximum speed of RTMPose, just for verification.
- Model file can be either a local path or a download link

```shell
# go to the mmpose folder
cd ${PATH_TO_MMPOSE}

# inference with rtmdet
python demo/topdown_demo_with_mmdet.py \
    projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input {YOUR_TEST_IMG_or_VIDEO} \
    --show

# inference with webcam
python demo/topdown_demo_with_mmdet.py \
    projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input webcam \
    --show
```

Result is as follows:

![topdown_inference_with_rtmdet](https://user-images.githubusercontent.com/13503330/220005020-06bdf37f-6817-4681-a2c8-9dd55e4fbf1e.png)

### Pre-compiled MMDeploy SDK (Recommended)

MMDeploy provides a precompiled SDK for Pipeline reasoning on RTMPose projects, where the model used for reasoning is the SDK version.

- All models must by exported by `tools/deploy.py` before PoseTracker can be used for inference.
- For the tutorial of exporting the SDK version model, see [SDK Reasoning](#%EF%B8%8F-step3-inference-with-sdk), and for detailed parameter settings of inference, see [Pipeline Reasoning](#-step4-pipeline-inference).
- Exported SDK models (ONNX, TRT, ncnn, etc.) can be downloaded from [OpenMMLab Deploee](https://platform.openmmlab.com/deploee).
- You can also convert `.pth` models into SDK [online](https://platform.openmmlab.com/deploee/task-convert-list).

#### Linux

Env Requirements:

- GCC >= 7.5
- cmake >= 3.20

##### Python Inference

1. Install mmdeploy_runtime or mmdeploy_runtime_gpu

```shell
# for onnxruntime
pip install mmdeploy-runtime

# for onnxruntime-gpu / tensorrt
pip install mmdeploy-runtime-gpu
```

2. Download Pre-compiled files.

```shell
# onnxruntime
# for ubuntu
wget -c  https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-linux-x86_64-cxx11abi.tar.gz
# unzip then add third party runtime libraries to the PATH

# for centos7 and lower
wget -c https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-linux-x86_64.tar.gz
# unzip then add third party runtime libraries to the PATH

# onnxruntime-gpu / tensorrt
# for ubuntu
wget -c  https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3.tar.gz
# unzip then add third party runtime libraries to the PATH

# for centos7 and lower
wget -c https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-linux-x86_64-cuda11.3.tar.gz
# unzip then add third party runtime libraries to the PATH
```

3. Download the sdk models and unzip to `./example/python`. (If you need other models, please export sdk models refer to [SDK Reasoning](#%EF%B8%8F-step3-inference-with-sdk))

```shell
# rtmdet-nano + rtmpose-m for cpu sdk
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-cpu.zip

unzip rtmpose-cpu.zip
```

4. Inference with `pose_tracker.py`:

```shell
# go to ./example/python

# Please pass the folder of the model, not the model file
# Format:
# python pose_tracker.py cpu {det work-dir} {pose work-dir} {your_video.mp4}

# Example:
python pose_tracker.py cpu rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ your_video.mp4

# webcam
python pose_tracker.py cpu rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ 0
```

##### ONNX

```shell
# Download pre-compiled files
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-linux-x86_64-cxx11abi.tar.gz

# Unzip files
tar -xzvf mmdeploy-1.0.0-linux-x86_64-cxx11abi.tar.gz

# Go to the sdk folder
cd mmdeploy-1.0.0-linux-x86_64-cxx11abi

# Init environment
source set_env.sh

# If opencv 3+ is not installed on your system, execute the following command.
# If it is installed, skip this command
bash install_opencv.sh

# Compile executable programs
bash build_sdk.sh

# Inference for an image
# Please pass the folder of the model, not the model file
./bin/det_pose rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ your_img.jpg --device cpu

# Inference for a video
# Please pass the folder of the model, not the model file
./bin/pose_tracker rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ your_video.mp4 --device cpu

# Inference using webcam
# Please pass the folder of the model, not the model file
./bin/pose_tracker rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ 0 --device cpu
```

##### TensorRT

```shell
# Download pre-compiled files
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3.tar.gz

# Unzip files
tar -xzvf mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3.tar.gz

# Go to the sdk folder
cd mmdeploy-1.0.0-linux-x86_64-cxx11abi-cuda11.3

# Init environment
source set_env.sh

# If opencv 3+ is not installed on your system, execute the following command.
# If it is installed, skip this command
bash install_opencv.sh

# Compile executable programs
bash build_sdk.sh

# Inference for an image
# Please pass the folder of the model, not the model file
./bin/det_pose rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ your_img.jpg --device cuda

# Inference for a video
# Please pass the folder of the model, not the model file
./bin/pose_tracker rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ your_video.mp4 --device cuda

# Inference using webcam
# Please pass the folder of the model, not the model file
./bin/pose_tracker rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ 0 --device cuda
```

For details, see [Pipeline Inference](#-step4-pipeline-inference).

#### Windows

##### Python Inference

1. Install mmdeploy_runtime or mmdeploy_runtime_gpu

```shell
# for onnxruntime
pip install mmdeploy-runtime
# download [sdk](https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-windows-amd64.zip) add third party runtime libraries to the PATH

# for onnxruntime-gpu / tensorrt
pip install mmdeploy-runtime-gpu
# download [sdk](https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0/mmdeploy-1.0.0-windows-amd64-cuda11.3.zip) add third party runtime libraries to the PATH
```

2. Download the sdk models and unzip to `./example/python`. (If you need other models, please export sdk models refer to [SDK Reasoning](#%EF%B8%8F-step3-inference-with-sdk))

```shell
# rtmdet-nano + rtmpose-m for cpu sdk
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-cpu.zip

unzip rtmpose-cpu.zip
```

3. Inference with `pose_tracker.py`:

```shell
# go to ./example/python
# Please pass the folder of the model, not the model file
python pose_tracker.py cpu rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ your_video.mp4

# Inference using webcam
# Please pass the folder of the model, not the model file
python pose_tracker.py cpu rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ 0
```

##### Executable Inference

1. Install [CMake](https://cmake.org/download/).
2. Download the [pre-compiled SDK](https://github.com/open-mmlab/mmdeploy/releases).
3. Unzip the SDK and go to the `sdk` folder.
4. open windows powerShell with administrator privileges

```shell
set-ExecutionPolicy RemoteSigned
```

5. Install OpenCV:

```shell
# in sdk folder:
.\install_opencv.ps1
```

6. Set environment variables:

```shell
# in sdk folder:
. .\set_env.ps1
```

7. Compile the SDK:

```shell
# in sdk folder:
# (if you installed opencv by .\install_opencv.ps1)
.\build_sdk.ps1
# (if you installed opencv yourself)
.\build_sdk.ps1 "path/to/folder/of/OpenCVConfig.cmake"
```

8. the executable will be generated in:

```shell
example\cpp\build\Release
```

### MMPose demo scripts

MMPose provides demo scripts to conduct [inference with existing models](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html).

**Note:**

- Inferencing with Pytorch can not reach the maximum speed of RTMPose, just for verification.

```shell
# go to the mmpose folder
cd ${PATH_TO_MMPOSE}

# inference with rtmdet
python demo/topdown_demo_with_mmdet.py \
    projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    {PATH_TO_CHECKPOINT}/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    {PATH_TO_CHECKPOINT}/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input {YOUR_TEST_IMG_or_VIDEO} \
    --show

# inference with webcam
python demo/topdown_demo_with_mmdet.py \
    projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    {PATH_TO_CHECKPOINT}/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    {PATH_TO_CHECKPOINT}/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input webcam \
    --show
```

Result is as follows:

![topdown_inference_with_rtmdet](https://user-images.githubusercontent.com/13503330/220005020-06bdf37f-6817-4681-a2c8-9dd55e4fbf1e.png)

## 👨‍🏫 How to Train [🔝](#-table-of-contents)

Please refer to [Train and Test](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html).

**Tips**:

- Please accordinally reduce `batch_size` and `base_lr` when your dataset is small.
- Guidelines to choose a model
  - m: Recommended and Preferred Use
  - t/s: For mobile devices with extremely low computing power, or scenarios with stringent inference speed requirements
  - l: Suitable for scenarios with strong computing power and not sensitive to speed

## 🏗️ How to Deploy [🔝](#-table-of-contents)

Here is a basic example of deploy RTMPose with [MMDeploy](https://github.com/open-mmlab/mmdeploy/tree/main).

- Exported SDK models (ONNX, TRT, ncnn, etc.) can be downloaded from [OpenMMLab Deploee](https://platform.openmmlab.com/deploee).
- You can also convert `.pth` models into SDK [online](https://platform.openmmlab.com/deploee/task-convert-list).

### 🧩 Step1. Install MMDeploy

Before starting the deployment, please make sure you install MMPose and MMDeploy correctly.

- Install MMPose, please refer to the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html).
- Install MMDeploy, please refer to the [MMDeploy installation guide](https://mmdeploy.readthedocs.io/en/latest/get_started.html#installation).

Depending on the deployment backend, some backends require compilation of custom operators, so please refer to the corresponding document to ensure the environment is built correctly according to your needs:

- [ONNX RUNTIME SUPPORT](https://mmdeploy.readthedocs.io/en/latest/05-supported-backends/onnxruntime.html)
- [TENSORRT SUPPORT](https://mmdeploy.readthedocs.io/en/latest/05-supported-backends/tensorrt.html)
- [OPENVINO SUPPORT](https://mmdeploy.readthedocs.io/en/latest/05-supported-backends/openvino.html)
- [More](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/05-supported-backends)

### 🛠️ Step2. Convert Model

After the installation, you can enjoy the model deployment journey starting from converting PyTorch model to backend model by running MMDeploy's `tools/deploy.py`.

The detailed model conversion tutorial please refer to the [MMDeploy document](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/convert_model.html). Here we only give the example of converting RTMPose.

Here we take converting RTMDet-nano and RTMPose-m to ONNX/TensorRT as an example.

- If you only want to use ONNX, please use:
  - [`detection_onnxruntime_static.py`](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmdet/detection/detection_onnxruntime_static.py) for RTMDet.
  - [`pose-detection_simcc_onnxruntime_dynamic.py`](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py) for RTMPose.
- If you want to use TensorRT, please use：
  - [`detection_tensorrt_static-320x320.py`](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmdet/detection/detection_tensorrt_static-320x320.py) for RTMDet.
  - [`pose-detection_simcc_tensorrt_dynamic-256x192.py`](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py) for RTMPose.

If you want to customize the settings in the deployment config for your requirements, please refer to [MMDeploy config tutorial](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/write_config.html).

In this tutorial, we organize files as follows:

```shell
|----mmdeploy
|----mmdetection
|----mmpose
```

#### ONNX

```shell
# go to the mmdeploy folder
cd ${PATH_TO_MMDEPLOY}

# run the command to convert RTMDet
# Model file can be either a local path or a download link
python tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_static.py \
    ../mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-ort/rtmdet-nano \
    --device cpu \
    --show \
    --dump-info  # dump sdk info

# run the command to convert RTMPose
# Model file can be either a local path or a download link
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-ort/rtmpose-m \
    --device cpu \
    --show \
    --dump-info  # dump sdk info
```

The converted model file is `{work-dir}/end2end.onnx` by defaults.

#### TensorRT

```shell
# go to the mmdeploy folder
cd ${PATH_TO_MMDEPLOY}

# run the command to convert RTMDet
# Model file can be either a local path or a download link
python tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_static-320x320.py \
    ../mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-trt/rtmdet-nano \
    --device cuda:0 \
    --show \
    --dump-info  # dump sdk info

# run the command to convert RTMPose
# Model file can be either a local path or a download link
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-trt/rtmpose-m \
    --device cuda:0 \
    --show \
    --dump-info  # dump sdk info
```

The converted model file is `{work-dir}/end2end.engine` by defaults.

🎊 If the script runs successfully, you will see the following files:

![convert_models](https://user-images.githubusercontent.com/13503330/217726963-7815dd01-561a-4605-b0c6-07b6fe1956c3.png)

#### Advanced Setting

To convert the model with TRT-FP16, you can enable the fp16 mode in your deploy config:

```Python
# in MMDeploy config
backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True  # enable fp16
    ))
```

### 🕹️ Step3. Inference with SDK

We provide both Python and C++ inference API with MMDeploy SDK.

To use SDK, you need to dump the required info during converting the model. Just add --dump-info to the model conversion command.

```shell
# RTMDet
# Model file can be either a local path or a download link
python tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_dynamic.py \
    ../mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-ort/rtmdet-nano \
    --device cpu \
    --show \
    --dump-info  # dump sdk info

# RTMPose
# Model file can be either a local path or a download link
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-ort/rtmpose-m \
    --device cpu \
    --show \
    --dump-info  # dump sdk info
```

After running the command, it will dump 3 json files additionally for the SDK:

```shell
|----{work-dir}
     |----end2end.onnx    # ONNX model
     |----end2end.engine  # TensorRT engine file

     |----pipeline.json   #
     |----deploy.json     # json files for the SDK
     |----detail.json     #
```

#### Python API

Here is a basic example of SDK Python API:

```Python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import numpy as np
from mmdeploy_runtime import PoseDetector


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

Here is a basic example of SDK C++ API:

```C++
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

To build C++ example, please add MMDeploy package in your CMake project as following:

```CMake
find_package(MMDeploy REQUIRED)
target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
```

#### Other languages

- [C# API Examples](https://github.com/open-mmlab/mmdeploy/tree/main/demo/csharp)
- [JAVA API Examples](https://github.com/open-mmlab/mmdeploy/tree/main/demo/java)

## 🚀 Step4. Pipeline Inference

### Inference for images

If the user has MMDeploy compiled correctly, you will see the `det_pose` executable under the `mmdeploy/build/bin/`.

```shell
# go to the mmdeploy folder
cd ${PATH_TO_MMDEPLOY}/build/bin/

# inference for an image
./det_pose rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ your_img.jpg --device cpu

required arguments:
  det_model           Object detection model path [string]
  pose_model          Pose estimation model path [string]
  image               Input image path [string]

optional arguments:
  --device            Device name, e.g. "cpu", "cuda" [string = "cpu"]
  --output            Output image path [string = "det_pose_output.jpg"]
  --skeleton          Path to skeleton data or name of predefined skeletons:
                      "coco" [string = "coco", "coco-wholoebody"]
  --det_label         Detection label use for pose estimation [int32 = 0]
                      (0 refers to 'person' in coco)
  --det_thr           Detection score threshold [double = 0.5]
  --det_min_bbox_size Detection minimum bbox size [double = -1]
  --pose_thr          Pose key-point threshold [double = 0]
```

#### API Example

- [`det_pose.py`](https://github.com/open-mmlab/mmdeploy/blob/main/demo/python/det_pose.py)
- [`det_pose.cxx`](https://github.com/open-mmlab/mmdeploy/blob/main/demo/csrc/cpp/det_pose.cxx)

### Inference for a video

If the user has MMDeploy compiled correctly, you will see the `pose_tracker` executable under the `mmdeploy/build/bin/`.

- pass `0` to `input` can inference from a webcam

```shell
# go to the mmdeploy folder
cd ${PATH_TO_MMDEPLOY}/build/bin/

# inference for a video
./pose_tracker rtmpose-ort/rtmdet-nano/ rtmpose-ort/rtmpose-m/ your_video.mp4 --device cpu

required arguments:
  det_model             Object detection model path [string]
  pose_model            Pose estimation model path [string]
  input                 Input video path or camera index [string]

optional arguments:
  --device              Device name, e.g. "cpu", "cuda" [string = "cpu"]
  --output              Output video path or format string [string = ""]
  --output_size         Long-edge of output frames [int32 = 0]
  --flip                Set to 1 for flipping the input horizontally [int32 = 0]
  --show                Delay passed to `cv::waitKey` when using `cv::imshow`;
                        -1: disable [int32 = 1]
  --skeleton            Path to skeleton data or name of predefined skeletons:
                        "coco", "coco-wholebody" [string = "coco"]
  --background          Output background, "default": original image, "black":
                        black background [string = "default"]
  --det_interval        Detection interval [int32 = 1]
  --det_label           Detection label use for pose estimation [int32 = 0]
                        (0 refers to 'person' in coco)
  --det_thr             Detection score threshold [double = 0.5]
  --det_min_bbox_size   Detection minimum bbox size [double = -1]
  --det_nms_thr         NMS IOU threshold for merging detected bboxes and
                        bboxes from tracked targets [double = 0.7]
  --pose_max_num_bboxes Max number of bboxes used for pose estimation per frame
                        [int32 = -1]
  --pose_kpt_thr        Threshold for visible key-points [double = 0.5]
  --pose_min_keypoints  Min number of key-points for valid poses, -1 indicates
                        ceil(n_kpts/2) [int32 = -1]
  --pose_bbox_scale     Scale for expanding key-points to bbox [double = 1.25]
  --pose_min_bbox_size  Min pose bbox size, tracks with bbox size smaller than
                        the threshold will be dropped [double = -1]
  --pose_nms_thr        NMS OKS/IOU threshold for suppressing overlapped poses,
                        useful when multiple pose estimations collapse to the
                        same target [double = 0.5]
  --track_iou_thr       IOU threshold for associating missing tracks
                        [double = 0.4]
  --track_max_missing   Max number of missing frames before a missing tracks is
                        removed [int32 = 10]
```

#### API Example

- [`pose_tracker.py`](https://github.com/open-mmlab/mmdeploy/blob/main/demo/python/pose_tracker.py)
- [`pose_tracker.cxx`](https://github.com/open-mmlab/mmdeploy/blob/main/demo/csrc/cpp/pose_tracker.cxx)

## 📚 Common Usage [🔝](#-table-of-contents)

### 🚀 Inference Speed Test [🔝](#-table-of-contents)

If you need to test the inference speed of the model under the deployment framework, MMDeploy provides a convenient `tools/profiler.py` script.

The user needs to prepare a folder for the test images `./test_images`, the profiler will randomly read images from this directory for the model speed test.

```shell
python tools/profiler.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    {RTMPOSE_PROJECT}/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../test_images \
    --model {WORK_DIR}/end2end.onnx \
    --shape 256x192 \
    --device cpu \
    --warmup 50 \
    --num-iter 200
```

The result is as follows:

```shell
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

If you want to learn more details of profiler, you can refer to the [Profiler Docs](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/useful_tools.html#profiler).

### 📊 Model Test [🔝](#-table-of-contents)

If you need to test the inference accuracy of the model on the deployment backend, MMDeploy provides a convenient `tools/test.py` script.

```shell
python tools/test.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    {RTMPOSE_PROJECT}/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    --model {PATH_TO_MODEL}/rtmpose_m.pth \
    --device cpu
```

You can also refer to [MMDeploy Docs](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/02-how-to-run/profile_model.md) for more details.

## 📜 Citation [🔝](#-table-of-contents)

If you find RTMPose useful in your research, please consider cite:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2303.07399,
  doi = {10.48550/ARXIV.2303.07399},
  url = {https://arxiv.org/abs/2303.07399},
  author = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
