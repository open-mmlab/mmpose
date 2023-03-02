# MMPose for AIGC

This project will demonstrate how to use MMPose to generate skeleton images for pose guided AI image generation.

## Get Started

English | [简体中文](./README_CN.md)

### Step 1: Preparation

#### Download Pre-compiled MMdeploy and Build

```shell
# Download pre-compiled files
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0rc3/mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz

# Unzip files
tar -xzvf mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz

# Go to the sdk folder
cd mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1/sdk

# Init environment
source env.sh

# If opencv 3+ is not installed on your system, execute the following command.
# If it is installed, skip this command
bash opencv.sh

# Compile executable programs
bash build.sh
```

#### Download Models

```shell
# Go to mmdeploy folder
cd ../

# Download models
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-cpu.zip

# Unzip files
unzip rtmpose-cpu.zip
```

The files are organized as follows:

```shell
|----mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1
|    |----sdk
|    |----rtmpose-ort
|    |    |----rtmdet-nano
|    |    |----rtmpose-m
|    |    |----000000147979.jpg
|    |    |----t2i-adapter_skeleton.txt
```

### Step 2: Generate a Skeleton Image

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

For details, you can refer to [here](../rtmpose/README.md).

The input image and its skeleton are as follows:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318807-a0a2a87a-cfc4-46cc-b647-e8f8bc44cfe3.jpg" width=450 /><img src='https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg' width=450/>
</div>

### Step 3: Upload to T2I-Adapter

The demo page of T2I- Adapter is [Here](https://huggingface.co/spaces/Adapter/T2I-Adapter).

[![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/ChongMou/T2I-Adapter)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319401-883daca0-ba99-4d21-850c-199aa7868e0f.png" width=900 />
</div>

For example:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319507-c8862ac3-43a9-4672-8f57-ae2f3c2834e6.png" width=900 />
</div>

## Gallery

> A lady with a fish

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319618-ee71ea71-88e2-4b61-81a0-bf675e0f8fbc.jpg" width=450 height=500 /><img src="https://user-images.githubusercontent.com/13503330/222319709-c19bef9a-ff02-4b09-a499-afd24f62399b.png" width=450 height=500/>
</div>

> An astronaut riding a bike on the moon

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319618-ee71ea71-88e2-4b61-81a0-bf675e0f8fbc.jpg" width=450 height=500 /><img src="https://user-images.githubusercontent.com/13503330/222319895-c753620e-9c02-49ea-8586-c021d50f7225.png" width=450 height=500/>
</div>

> An astronaut riding a bike on Mars

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319618-ee71ea71-88e2-4b61-81a0-bf675e0f8fbc.jpg" width=450 height=500 /><img src="https://user-images.githubusercontent.com/13503330/222319949-e4b4f5ff-888e-4080-9c1e-9bcafd17f306.png" width=450 height=500/>
</div>

> An astronaut riding a bike on Jupiter

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222319618-ee71ea71-88e2-4b61-81a0-bf675e0f8fbc.jpg" width=450 height=500 /><img src="https://user-images.githubusercontent.com/13503330/222320017-ae7fe863-fda6-4dcc-b9fb-2a44609c170f.png" width=450 height=500/>
</div>
