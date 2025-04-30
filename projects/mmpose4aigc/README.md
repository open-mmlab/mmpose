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

### Generate OpenPose-style Skeleton

#### Step 1: Preparation

Run the following commands to prepare the project:

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

#### Step 2: Generate a Skeleton Image

Run the following command to generate a skeleton image:

```shell
# generate a skeleton image
bash mmpose_openpose.sh ../../tests/data/coco/000000000785.jpg
```

The input image and its skeleton are as follows:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/226527894-f9d98e75-fc6c-49e5-ba39-d6277b03697a.jpg" width=450 /><img src='https://user-images.githubusercontent.com/13503330/226527555-58a86ee6-a886-4986-b1c3-b3841b112995.png' width=450/>
</div>

### Generate MMPose-style Skeleton

#### Step 1: Preparation

**Env Requirements:**

- GCC >= 7.5
- cmake >= 3.14

Run the following commands to install the project:

```shell
bash install_posetracker_linux.sh
```

After installation, files are organized as follows:

```shell
|----mmdeploy-1.0.0-linux-x86_64-cxx11abi
|    |----README.md
|    |----rtmpose-ort
|    |    |----rtmdet-nano
|    |    |----rtmpose-m
|    |    |----000000147979.jpg
|    |    |----t2i-adapter_skeleton.txt
```

#### Step 2: Generate a Skeleton Image

Run the following command to generate a skeleton image:

```shell
# generate a skeleton image
bash mmpose_style_skeleton.sh \
    mmdeploy-1.0.0-linux-x86_64-cxx11abi/rtmpose-ort/000000147979.jpg
```

For more details, you can refer to [RTMPose](../rtmpose/README.md).

The input image and its skeleton are as follows:

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/222318807-a0a2a87a-cfc4-46cc-b647-e8f8bc44cfe3.jpg" width=300 /><img src='https://user-images.githubusercontent.com/13503330/222318943-6dba5f52-158a-427a-8222-03628addc051.jpg' width=300/>
</div>

### Upload to T2I-Adapter

The demo page of T2I- Adapter is [Here](https://huggingface.co/spaces/Adapter/T2I-Adapter).

[![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/ChongMou/T2I-Adapter)

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/226528354-aac50e26-9188-4b81-9692-274415ee5a87.png" width=900 />
</div>

## Gallery

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/226529427-4a34f168-caa2-446b-8aa9-2a60c4082921.png" width=900 />
</div>
