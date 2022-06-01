<!-- TOC -->

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Best Practices](#best-practices)
  - [Verify the installation](#verify-the-installation)
  - [Customize Installation](#customize-installation)
    - [CUDA versions](#cuda-versions)
    - [Install MMCV without MIM](#install-mmcv-without-mim)
    - [Install on CPU-only platforms](#install-on-cpu-only-platforms)
    - [Install on Google Colab](#install-on-google-colab)
    - [Using MMPose with Docker](#using-mmpose-with-docker)
  - [Trouble shooting](#trouble-shooting)

<!-- TOC -->

# Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMPose works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.

If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

```{warning}
This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Installation

We recommend that users follow our best practices to install MMPose. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

## Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMPose.

Case a: If you develop and run mmpose directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmpose as a dependency or third-party package, install it with pip:

```shell
pip install mmpose
```

## Verify the installation

To verify that MMPose is installed correctly, you can run an inference demo with the following steps.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `associative_embedding_hrnet_w32_coco_512x512.py` and `hrnet_w32_coco_512x512-bcb8c247_20200816.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install mmpose from source, just run the following command.

```shell
python demo/bottom_up_img_demo.py associative_embedding_hrnet_w32_coco_512x512.py hrnet_w32_coco_512x512-bcb8c247_20200816.pth --img-path tests/data/coco/ --out-img-root vis_results
```

You will see several images in this folder: `vis_results`, where the human pose estimation results are plotted on the images.

Option (b). If you install mmpose with pip, open you python interpreter and copy&paste the following codes.

```python
from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)

config_file = 'associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = 'hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

image_name = 'demo/persons.jpg'
# test a single image
pose_results, _ = inference_bottom_up_pose_model(pose_model, image_name)

# show the results
vis_pose_result(pose_model, image_name, pose_results, out_file='demo/vis_persons.jpg')
```

Prepare an image with persons and place it properly, then run the above code, and you will see the output image with detected human poses ploted on it.

## Customize Installation

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Install on CPU-only platforms

MMPose can be built for CPU only environment. In CPU mode you can train (requires MMCV version >= 1.4.4), test or inference a model.

However, some functionalities are missing in this mode, usually GPU-compiled ops like `Deformable Convolution`. Most models in MMPose don't depend on these ops, but if you try to train/test/infer a model containing these ops, an error will be raised.

### Install on Google Colab

[Google Colab](https://colab.research.google.com/) usually has PyTorch installed,
thus we only need to install MMCV and MMPose with the following commands.

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmcv-full
```

**Step 2.** Install MMPose from the source.

```shell
!git clone https://github.com/open-mmlab/mmpose.git
%cd mmpose
!pip install -e .
```

**Step 3.** Verification.

```python
import mmpose
print(mmpose.__version__)
# Example output: 0.26.0
```

```{note}
Note that within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

### Using MMPose with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmpose/blob/master/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.6.0, CUDA 10.1, CUDNN 7.
# If you prefer other versions, just modified the Dockerfile
docker build -t mmpose docker/
```

**Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmpose/data mmpose
```

`{DATA_DIR}` is your local folder containing all the datasets for mmpose.

```{note}
If you encounter the error message like `permission denied`, please add `sudo` at the start of the command and try it again.
```

## Trouble shooting

If you have some issues during the installation, please first view the [FAQ](faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmpose/issues/new/choose) on GitHub if no solution is found.
