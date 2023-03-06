# MMPose for AIGC (AI Generated Content)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222403836-c65ba905-4bdd-4a44-834c-ff8d5959649d.png" width=1000 />
</div>

简体中文 | [English](./README.md)

本项目将支持使用 MMPose 来生成骨架图片，用于姿态引导的 AI 图像生成。

当前已支持：

- [T2I Adapter](https://huggingface.co/spaces/Adapter/T2I-Adapter)

欢迎分享更多姿态引导的 AIGC 项目给我们！

## 快速上手

### Step 1: 准备

#### 安装环境

```shell
# 下载预编译包
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0rc3/mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz

# 解压
tar -xzvf mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz

# 切换目录
cd mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1/sdk

# 初始化环境
source env.sh

# 安装 OpenCV
bash opencv.sh

# 本地编译
bash build.sh
```

#### 下载模型

```shell
# 切换目录
cd ../

# 下载模型
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-cpu.zip

# 解压
unzip rtmpose-cpu.zip
```

最终的文件结构如下：

```shell
|----mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1
|    |----sdk
|    |----rtmpose-ort
|    |    |----rtmdet-nano
|    |    |----rtmpose-m
|    |    |----000000147979.jpg
|    |    |----t2i-adapter_skeleton.txt
```

### Step 2: 生成姿态骨架图片

```shell
./bin/pose_tracker \
    ./rtmpose-ort/rtmdet-nano/ \
    ./rtmpose-ort/rtmpose-m \
    ./rtmpose-ort/000000147979.jpg \
    --background black \
    --skeleton ./rtmpose-ort/t2i-adapter_skeleton.txt \
    --output ./skeleton_res.jpg \
    --pose_kpt_thr 0.4 \
    --show -1
```

**参数说明：**

```shell
required arguments:
  det_model             Detection 模型路径 [string]
  pose_model            Pose 模型路径 [string]
  input                 输入图片路径或摄像头序号 [string]

optional arguments:
  --device              推理设备 "cpu", "cuda" [string = "cpu"]
  --output              导出视频路径 [string = ""]
  --output_size         输出视频帧的长边 [int32 = 0]
  --flip                设置为1，用于水平翻转输入 [int32 = 0]
  --show                使用`cv::imshow`时，传递给`cv::waitKey`的延迟;
                        -1: 关闭 [int32 = 1]
  --skeleton            骨架数据的路径或预定义骨架的名称:
                        "coco", "coco-wholebody" [string = "coco"]
  --background          导出视频背景颜色, "default": 原图, "black":
                        纯黑背景 [string = "default"]
  --det_interval        检测间隔 [int32 = 1]
  --det_label           用于姿势估计的检测标签 [int32 = 0]
                        (0 在 coco 中对应 person)
  --det_thr             检测分数阈值 [double = 0.5]
  --det_min_bbox_size   最小检测框大小 [double = -1]
  --det_nms_thr         NMS IOU阈值，用于合并检测到的bboxes和
                        追踪到的目标的 bboxes [double = 0.7]
  --pose_max_num_bboxes 每一帧用于姿势估计的 bboxes 的最大数量
                        [int32 = -1]
  --pose_kpt_thr        可见关键点的阈值 [double = 0.5]
  --pose_min_keypoints  有效姿势的最小关键点数量，-1表示上限(n_kpts/2) [int32 = -1]
  --pose_bbox_scale     将关键点扩展到 bbox 的比例 [double = 1.25]
  --pose_min_bbox_size  最小追踪尺寸，尺寸小于阈值的 bbox 将被剔除 [double = -1]
  --pose_nms_thr        用于抑制重叠姿势的 NMS OKS/IOU阈值。
                        当多个姿态估计重叠到同一目标时非常有用 [double = 0.5]
  --track_iou_thr       追踪 IOU 阈值 [double = 0.4]
  --track_max_missing   最大追踪容错 [int32 = 10]
```

更多详细信息可以查看 [RTMPose](../rtmpose/README_CN.md)。

输入图片与生成骨架图片如下:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318807-a0a2a87a-cfc4-46cc-b647-e8f8bc44cfe3.jpg" width=450 /><img src='https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg' width=450/>
</div>

### Step 3: 使用 T2I-Adapter

T2I- Adapter 在线试玩请点击 [这里](https://huggingface.co/spaces/Adapter/T2I-Adapter)

[![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/ChongMou/T2I-Adapter)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319401-883daca0-ba99-4d21-850c-199aa7868e0f.png" width=900 />
</div>

示例：

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319507-c8862ac3-43a9-4672-8f57-ae2f3c2834e6.png" width=900 />
</div>

## 结果展示

> A lady with a fish

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319618-ee71ea71-88e2-4b61-81a0-bf675e0f8fbc.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222319709-c19bef9a-ff02-4b09-a499-afd24f62399b.png" width=280 height=300/>
</div>

> An astronaut riding a bike on the moon

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222319895-c753620e-9c02-49ea-8586-c021d50f7225.png" width=280 height=300/>
</div>

> An astronaut riding a bike on Mars

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222319949-e4b4f5ff-888e-4080-9c1e-9bcafd17f306.png" width=280 height=300/>
</div>

> An astronaut riding a bike on Jupiter

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222320017-ae7fe863-fda6-4dcc-b9fb-2a44609c170f.png" width=280 height=300/>
</div>

> Monkey king

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg" width=280 height=300 /><img src="https://user-images.githubusercontent.com/13503330/222341871-4beac696-7d51-490b-94b2-2e3f1adb6927.jpg" width=280 height=300/>
</div>
