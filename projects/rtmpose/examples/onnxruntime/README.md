# RTMPose inference with ONNXRuntime

This example shows how to run RTMPose inference with ONNXRuntime in Python.

## Prerequisites

### 1. Install onnxruntime inference engine.

Choose one of the following ways to install onnxruntime.

- CPU version

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

- GPU version

```bash
pip install onnxruntime-gpu==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-gpu-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-gpu-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

### 2. Convert model to onnx files

- Install `mim` tool.

```bash
pip install -U openmim
```

- Download `mmpose` model.

```bash
# choose one rtmpose model
mim download mmpose --config rtmpose-m_8xb64-270e_coco-wholebody-256x192 --dest .
```

- Clone `mmdeploy` repo.

```bash
git clone https://github.com/open-mmlab/mmdeploy.git
```

- Convert model to onnx files.

```bash
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    mmpose/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py \
    mmpose/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth \
    mmdeploy/demo/resources/human-pose.jpg \
    --work-dir mmdeploy_model/mmpose/ort \
    --device cuda \
    --dump-info
```

## Run demo

### Install dependencies

```bash
pip install -r requirements.txt
```

### Usage:

```bash
python main.py \
    {ONNX_FILE} \
    {IMAGE_FILE} \
    --device {DEVICE} \
    --save-path {SAVE_PATH}
```

### Description of all arguments

- `ONNX_FILE`: The path of onnx file
- `IMAGE_FILE`: The path of image file
- `DEVICE`: The device to run the model, default is `cpu`
- `SAVE_PATH`: The path to save the output image, default is `output.jpg`
