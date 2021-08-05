# Tutorial 5: Exporting a model to ONNX

Open Neural Network Exchange [(ONNX)](https://onnx.ai/) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves.

<!-- TOC -->

- [Supported Models](#supported-models)
- [Usage](#usage)
  - [Prerequisite](#prerequisite)

<!-- TOC -->

## Supported Models

So far, our codebase supports onnx exporting from pytorch models trained with MMPose. The supported models include:

- ResNet
- HRNet
- HigherHRNet

## Usage

For simple exporting, you can use the [script](/tools/pytorch2onnx.py) here. Note that the package `onnx` and `onnxruntime` are required for verification after exporting.

### Prerequisite

First, install onnx.

```shell
pip install onnx onnxruntime
```

We provide a python script to export the pytorch model trained by MMPose to ONNX.

```shell
python tools/deployment/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--shape ${SHAPE}] \
    [--verify] [--show] [--output-file ${OUTPUT_FILE}] [--opset-version ${VERSION}]
```

Optional arguments:

- `--shape`: The shape of input tensor to the model. If not specified, it will be set to `1 3 256 192`.
- `--verify`: Determines whether to verify the exported model, runnably and numerically. If not specified, it will be set to `False`.
- `--show`: Determines whether to print the architecture of the exported model. If not specified, it will be set to `False`.
- `--output-file`: The output onnx model name. If not specified, it will be set to `tmp.onnx`.
- `--opset-version`: Determines the operation set version of onnx, we recommend you to use a higher version such as 11 for compatibility. If not specified, it will be set to `11`.

Please fire an issue if you discover any checkpoints that are not perfectly exported or suffer some loss in accuracy.
