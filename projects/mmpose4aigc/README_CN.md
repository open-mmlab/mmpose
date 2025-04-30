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

### 生成 Openpose 风格的骨架图片

#### Step 1: 准备

运行以下命令准备项目：

```shell
# install mmpose mmdet
pip install openmim
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
mim install -e .
mim install "mmdet>=3.0.0rc6"

# download models
bash download_models.sh
```

#### Step 2: 生成骨架图片

运行以下命令生成骨架图片：

```shell
bash mmpose_openpose.sh ../../tests/data/coco/000000000785.jpg
```

输入图片与生成骨架图片如下:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/226527894-f9d98e75-fc6c-49e5-ba39-d6277b03697a.jpg" width=450 /><img src='https://user-images.githubusercontent.com/13503330/226527555-58a86ee6-a886-4986-b1c3-b3841b112995.png' width=450/>
</div>

### 生成 MMPose 风格的骨架图片

#### Step 1: 准备

**环境要求：**

- GCC >= 7.5
- cmake >= 3.14

运行以下命令安装项目：

```shell
bash install_posetracker_linux.sh
```

最终的文件结构如下：

```shell
|----mmdeploy-1.0.0-linux-x86_64-cxx11abi
|    |----README.md
|    |----rtmpose-ort
|    |    |----rtmdet-nano
|    |    |----rtmpose-m
|    |    |----000000147979.jpg
|    |    |----t2i-adapter_skeleton.txt
```

#### Step 2: 生成姿态骨架图片

运行以下命令生成姿态骨架图片：

```shell
# 生成骨架图片
bash mmpose_style_skeleton.sh \
    mmdeploy-1.0.0-linux-x86_64-cxx11abi/rtmpose-ort/000000147979.jpg
```

更多详细信息可以查看 [RTMPose](../rtmpose/README_CN.md)。

输入图片与生成骨架图片如下:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318807-a0a2a87a-cfc4-46cc-b647-e8f8bc44cfe3.jpg" width=450 /><img src='https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg' width=450/>
</div>

### 使用 T2I-Adapter

T2I- Adapter 在线试玩请点击 [这里](https://huggingface.co/spaces/Adapter/T2I-Adapter)

[![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/ChongMou/T2I-Adapter)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/226528354-aac50e26-9188-4b81-9692-274415ee5a87.png" width=900 />
</div>

## 结果展示

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/226529427-4a34f168-caa2-446b-8aa9-2a60c4082921.png" width=900 />
</div>
