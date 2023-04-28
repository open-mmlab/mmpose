# 使用ONNXRuntime进行RTMPose推理

本示例展示了如何在Python中用ONNXRuntime推理RTMPose模型。

## 准备

### 1. 安装onnxruntime推理引擎.

选择以下方式之一来安装onnxruntime。

- CPU版本

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

- GPU版本

```bash
pip install onnxruntime-gpu==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-gpu-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-gpu-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

### 2. 将模型转换为onnx文件

- 安装`mim`工具

```bash
pip install -U openmim
```

- 下载`mmpose`模型

```bash
# choose one rtmpose model
mim download mmpose --config rtmpose-m_8xb64-270e_coco-wholebody-256x192 --dest .
```

- 克隆`mmdeploy`仓库

```bash
git clone https://github.com/open-mmlab/mmdeploy.git
```

- 将模型转换为onnx文件

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

## 运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 用法：

```bash
python main.py \
    {ONNX_FILE} \
    {IMAGE_FILE} \
    --device {DEVICE} \
    --save-path {SAVE_PATH}
```

### 参数解释

- `ONNX_FILE`: onnx文件的路径
- `IMAGE_FILE`: 图像文件的路径
- `DEVICE`: 运行模型的设备，默认为\`cpu'
- `SAVE_PATH`: 保存输出图像的路径，默认为 "output.jpg"
