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

**环境要求：**

- GCC >= 7.5
- cmake >= 3.14

运行以下命令安装项目：

```shell
bash install.sh
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

运行以下命令生成姿态骨架图片：

```shell
bash mmpose_t2i-adapter.sh rtmpose-ort/000000147979.jpg
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
