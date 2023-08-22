# 模型精简与部署

本章将介绍如何导出与部署 MMPose 训练得到的模型，包含以下内容：

- [模型精简](#模型精简)
- [使用 MMDeploy 部署](#使用-mmdeploy-部署)
  - [MMDeploy 介绍](#mmdeploy-介绍)
  - [模型支持列表](#模型支持列表)
  - [安装](#安装)
  - [模型转换](#模型转换)
    - [如何查找 MMPose 模型对应的部署配置文件](#如何查找-mmpose-模型对应的部署配置文件)
    - [RTMPose 模型导出示例](#rtmpose-模型导出示例)
    - [ONNX](#onnx)
    - [TensorRT](#tensorrt)
    - [高级设置](#高级设置)
  - [模型测速](#模型测速)
  - [精度验证](#精度验证)

## 模型精简

在默认状态下，MMPose 训练过程中保存的 checkpoint 文件包含了模型的所有信息，包括模型结构、权重、优化器状态等。这些信息对于模型的部署来说是冗余的，因此我们需要对模型进行精简，精简后的 `.pth` 文件大小甚至能够缩小一半以上。

MMPose 提供了 [tools/misc/publish_model.py](https://github.com/open-mmlab/mmpose/blob/dev-1.x/tools/misc/publish_model.py) 来进行模型精简，使用方式如下：

```shell
python tools/misc/publish_model.py ${IN_FILE} ${OUT_FILE}
```

例如：

```shell
python tools/misc/publish_model.py ./epoch_10.pth ./epoch_10_publish.pth
```

脚本会自动对模型进行精简，并将精简后的模型保存到制定路径，并在文件名的最后加上时间戳，例如 `./epoch_10_publish-21815b2c_20230726.pth`。

## 使用 MMDeploy 部署

### MMDeploy 介绍

MMDeploy 是 OpenMMLab 模型部署工具箱，为各算法库提供统一的部署体验。基于 MMDeploy，开发者可以轻松从 MMPose 生成指定硬件所需 SDK，省去大量适配时间。

- 你可以从 [【硬件模型库】](https://platform.openmmlab.com/deploee) 直接下载 SDK 版模型（ONNX、TensorRT、ncnn 等）。
- 同时我们也支持 [在线模型转换](https://platform.openmmlab.com/deploee/task-convert-list)，从而无需本地安装 MMDeploy。

更多介绍和使用指南见 [MMDeploy 文档](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html)。

### 模型支持列表

| Model                                                                                                     | Task          | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO | CoreML | TorchScript |
| :-------------------------------------------------------------------------------------------------------- | :------------ | :----------: | :------: | :--: | :---: | :------: | :----: | :---------: |
| [HRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hrnet-cvpr-2019)          | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   Y    |      Y      |
| [MSPN](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#mspn-arxiv-2019)           | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   Y    |      Y      |
| [LiteHRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#litehrnet-cvpr-2021)  | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   Y    |      Y      |
| [Hourglass](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#hourglass-eccv-2016) | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   Y    |      Y      |
| [SimCC](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#simcc-eccv-2022)         | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   Y    |      Y      |
| [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)                                | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   Y    |      Y      |
| [YoloX-Pose](https://github.com/open-mmlab/mmpose/tree/main/projects/yolox_pose)                          | PoseDetection |      Y       |    Y     |  N   |   N   |    Y     |   Y    |      Y      |

### 安装

在开始部署之前，首先你需要确保正确安装了 MMPose, MMDetection, MMDeploy，相关安装教程如下：

- [安装 MMPose 与 MMDetection](../installation.md)
- [安装 MMDeploy](https://mmdeploy.readthedocs.io/zh_CN/latest/04-supported-codebases/mmpose.html)

根据部署后端的不同，有的后端需要对 MMDeploy 支持的**自定义算子进行编译**，请根据需求前往对应的文档确保环境搭建正确：

- [ONNX](https://mmdeploy.readthedocs.io/zh_CN/latest/05-supported-backends/onnxruntime.html)
- [TensorRT](https://mmdeploy.readthedocs.io/zh_CN/latest/05-supported-backends/tensorrt.html)
- [OpenVINO](https://mmdeploy.readthedocs.io/zh_CN/latest/05-supported-backends/openvino.html)
- [ncnn](https://mmdeploy.readthedocs.io/zh_CN/latest/05-supported-backends/ncnn.html)
- [TorchScript](https://mmdeploy.readthedocs.io/en/latest/05-supported-backends/torchscript.html)
- [更多](https://github.com/open-mmlab/mmdeploy/tree/main/docs/zh_cn/05-supported-backends)

### 模型转换

在完成安装之后，你就可以开始模型部署了。通过 MMDeploy 提供的 [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/blob/main/tools/deploy.py) 可以方便地将 MMPose 模型转换到不同的部署后端。

使用方法如下：

```shell
python ./tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --test-img ${TEST_IMG} \
    --work-dir ${WORK_DIR} \
    --calib-dataset-cfg ${CALIB_DATA_CFG} \
    --device ${DEVICE} \
    --log-level INFO \
    --show \
    --dump-info
```

参数描述：

- `deploy_cfg` : mmdeploy 针对此模型的部署配置，包含推理框架类型、是否量化、输入 shape 是否动态等。配置文件之间可能有引用关系，`configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py` 是一个示例。

- `model_cfg` : mm 算法库的模型配置，例如 `mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_aic-coco-256x192.py`，与 mmdeploy 的路径无关。

- `checkpoint` : torch 模型路径。可以 http/https 开头，详见 mmcv.FileClient 的实现。

- `img` : 模型转换时，用做测试的图像或点云文件路径。

- `--test-img` : 用于测试模型的图像文件路径。默认设置成None。

- `--work-dir` : 工作目录，用来保存日志和模型文件。

- `--calib-dataset-cfg` : 此参数只有int8模式下生效，用于校准数据集配置文件。若在int8模式下未传入参数，则会自动使用模型配置文件中的’val’数据集进行校准。

- `--device` : 用于模型转换的设备。 默认是cpu，对于 trt 可使用 cuda:0 这种形式。

- `--log-level` : 设置日记的等级，选项包括'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'。 默认是INFO。

- `--show` : 是否显示检测的结果。

- `--dump-info` : 是否输出 SDK 信息。

#### 如何查找 MMPose 模型对应的部署配置文件

1. 所有与 MMPose 相关的部署配置文件都存放在 [configs/mmpose/](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpose) 目录下。
2. 部署配置文件命名遵循 `{任务}_{算法}_{部署后端}_{动态/静态}_{输入尺寸}` 。

#### RTMPose 模型导出示例

我们本节演示将 RTMPose 模型导出为 ONNX 和 TensorRT 格式，如果你希望了解更多内容请前往 [MMDeploy 文档](https://mmdeploy.readthedocs.io/zh_CN/latest/02-how-to-run/convert_model.html)。

- ONNX 配置

  - [pose-detection_simcc_onnxruntime_dynamic.py](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py)

- TensorRT 配置

  - [pose-detection_simcc_tensorrt_dynamic-256x192.py](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py)

- 更多

  |  Backend  |                                                                                Config                                                                                |
  | :-------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
  | ncnn-fp16 | [pose-detection_simcc_ncnn-fp16_static-256x192.py](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmpose/pose-detection_simcc_ncnn-fp16_static-256x192.py) |
  |  CoreML   |    [pose-detection_simcc_coreml_static-256x192.py](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmpose/pose-detection_simcc_coreml_static-256x192.py)    |
  | OpenVINO  |  [pose-detection_simcc_openvino_static-256x192.py](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmpose/pose-detection_simcc_openvino_static-256x192.py)  |
  |   RKNN    | [pose-detection_simcc_rknn-fp16_static-256x192.py](https://github.com/open-mmlab/mmdeploy/blob/main/configs/mmpose/pose-detection_simcc_rknn-fp16_static-256x192.py) |

如果你需要对部署配置进行修改，请参考 [MMDeploy config tutorial](https://mmdeploy.readthedocs.io/zh_CN/latest/02-how-to-run/write_config.html).

本教程中使用的文件结构如下：

```shell
|----mmdeploy
|----mmpose
```

##### ONNX

运行如下命令：

```shell
# 前往 mmdeploy 目录
cd ${PATH_TO_MMDEPLOY}

# 转换 RTMPose
# 输入模型路径可以是本地路径，也可以是下载链接。
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-ort/rtmpose-m \
    --device cpu \
    --show \
    --dump-info   # 导出 sdk info
```

默认导出模型文件为 `{work-dir}/end2end.onnx`

##### TensorRT

运行如下命令：

```shell
# 前往 mmdeploy 目录
cd ${PATH_TO_MMDEPLOY}

# 转换 RTMPose
# 输入模型路径可以是本地路径，也可以是下载链接。
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-trt/rtmpose-m \
    --device cuda:0 \
    --show \
    --dump-info   # 导出 sdk info
```

默认导出模型文件为 `{work-dir}/end2end.engine`

如果模型顺利导出，你将会看到样例图片上的检测结果：

![convert_models](https://user-images.githubusercontent.com/13503330/217726963-7815dd01-561a-4605-b0c6-07b6fe1956c3.png)

###### 高级设置

如果需要使用 TensorRT-FP16，你可以通过修改 MMDeploy config 中以下配置开启：

```Python
# in MMDeploy config
backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True  # 打开 fp16
    ))
```

### 模型测速

如果需要测试模型在部署框架下的推理速度，MMDeploy 提供了方便的 [tools/profiler.py](https://github.com/open-mmlab/mmdeploy/blob/main/tools/profiler.py) 脚本。

用户需要准备一个存放测试图片的文件夹`./test_images`，profiler 将随机从该目录下抽取图片用于模型测速。

```shell
# 前往 mmdeploy 目录
cd ${PATH_TO_MMDEPLOY}

python tools/profiler.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    ../test_images \
    --model {WORK_DIR}/end2end.onnx \
    --shape 256x192 \
    --device cpu \
    --warmup 50 \
    --num-iter 200
```

测试结果如下：

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

```{note}
如果你希望详细了解 profiler 的更多参数设置与功能，可以前往 [Profiler 文档](https://mmdeploy.readthedocs.io/en/main/02-how-to-run/useful_tools.html#profiler)。
```

### 精度验证

如果需要测试模型在部署框架下的推理精度，MMDeploy 提供了方便的 `tools/test.py` 脚本。

```shell
# 前往 mmdeploy 目录
cd ${PATH_TO_MMDEPLOY}

python tools/test.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    ./mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    --model {PATH_TO_MODEL}/rtmpose_m.pth \
    --device cpu
```

```{note}
详细内容请参考 [MMDeploy 文档](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/02-how-to-run/profile_model.md)
```
