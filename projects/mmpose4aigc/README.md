# MMPose for AIGC (AI Generated Content)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222403836-c65ba905-4bdd-4a44-834c-ff8d5959649d.png" width=1000 />
</div>

English | [简体中文](./README_CN.md)

This project will demonstrate how to use MMPose to generate skeleton images for pose guided AI image generation.

Currently, we support:

- [T2I Adapter](https://huggingface.co/spaces/Adapter/T2I-Adapter)

Please feel free to share interesting pose-guided AIGC projects to us!

## Get Started

### Step 1: Preparation

**Env Requirements:**

- GCC >= 7.5
- cmake >= 3.14

Run following commands to install the project:

```shell
bash install_linux.sh
```

After installation, files are organized as follows:

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

Run following command to generate a skeleton image:

```shell
# generate a skeleton image
bash mmpose_t2i-adapter.sh \
    mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1/rtmpose-ort/000000147979.jpg
```

For more details, you can refer to [RTMPose](../rtmpose/README.md).

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
