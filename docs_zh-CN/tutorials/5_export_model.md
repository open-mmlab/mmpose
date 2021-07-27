# 教程 5：如何导出模型为 onnx 格式

开放式神经网络交换格式（Open Neural Network Exchange，即 [ONNX](https://onnx.ai/)）是各种框架共用的一种模型交换格式，AI 开发人员可以方便将模型部署到所需的框架之中。

<!-- TOC -->

- [支持的模型](#支持的模型)
- [如何使用](#如何使用)
  - [准备工作](#准备工作)

<!-- TOC -->

## 支持的模型

MMPose 支持将训练好的各种 Pytorch 模型导出为 ONNX 格式。支持的模型包括但不限于：

- ResNet
- HRNet
- HigherHRNet

## 如何使用

用户可以使用这里的 [脚本](/tools/deployment/pytorch2onnx.py) 来导出 ONNX 格式。

### 准备工作

首先，安装 onnx

```shell
pip install onnx onnxruntime
```

MMPose 提供了一个 python 脚本，将 MMPose 训练的 pytorch 模型导出到 ONNX。

```shell
python tools/deployment/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--shape ${SHAPE}] \
    [--verify] [--show] [--output-file ${OUTPUT_FILE}]  [--is-localizer] [--opset-version ${VERSION}]
```

可选参数：

- `--shape`: 模型输入张量的形状。对于 2D 关键点检测模型（如 HRNet），输入形状应当为 `$batch $channel $height $width` (例如，`1 3 256 192`)；
- `--verify`: 是否对导出模型进行验证，验证项包括是否可运行，数值是否正确等。如果没有手动指定，默认为 `False`。
- `--show`: 是否打印导出模型的结构。如果没有手动指定，默认为 `False`。
- `--output-file`: 导出的 onnx 模型名。如果没有手动指定，默认为 `tmp.onnx`。
- `--opset-version`：决定 onnx 的执行版本，MMPose 推荐用户使用高版本（例如 11 版本）的 onnx 以确保稳定性。如果没有手动指定，默认为 `11`。

如果发现提供的模型权重文件没有被成功导出，或者存在精度损失，可以在本 repo 下提出问题（issue）。
