# Installation

We recommend that users follow our best practices to install MMPose. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Best Practices](#best-practices)
    - [Build MMPose from source](#build-mmpose-from-source)
    - [Install as a Python package](#install-as-a-python-package)
  - [Customize Installation](#customize-installation)
    - [CUDA versions](#cuda-versions)
    - [Install MMEngine without MIM](#install-mmengine-without-mim)
    - [Install MMCV without MIM](#install-mmcv-without-mim)
    - [Install on CPU-only platforms](#install-on-cpu-only-platforms)
    - [Install on Google Colab](#install-on-google-colab)
    - [Using MMPose with Docker](#using-mmpose-with-docker)
  - [Verify the installation](#verify-the-installation)
  - [Trouble shooting](#trouble-shooting)

<!-- TOC -->

## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMPose works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.8+.

If you are experienced with PyTorch and have already installed it, you can skip this part and jump to the [MMPose Installation](#install-mmpose). Otherwise, you can follow these steps for the preparation.

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

**Step 3.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```

Note that some of the demo scripts in MMPose require [MMDetection](https://github.com/open-mmlab/mmdetection) (mmdet)  for human detection. If you want to run these demo scripts with mmdet, you can easily install mmdet as a dependency by running:

```shell
mim install "mmdet>=3.1.0"
```

## Best Practices

### Build MMPose from source

To develop and run mmpose directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### Install as a Python package

To use mmpose as a dependency or third-party package, install it with pip:

```shell
mim install "mmpose>=1.1.0"
```

## Verify the installation

To verify that MMPose is installed correctly, you can run an inference demo with the following steps.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `td-hm_hrnet-w48_8xb32-210e_coco-256x192.py` and `hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth` in your current folder.

**Step 2.** Run the inference demo.

Option (A). If you install mmpose from source, just run the following command under the folder `$MMPOSE`:

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --out-file vis_results.jpg \
    --draw-heatmap
```

If everything goes fine, you will be able to get the following visualization result from `vis_results.jpg` in your current folder, which displays the predicted keypoints and heatmaps overlaid on the person in the image.

![image](https://user-images.githubusercontent.com/87690686/187824033-2cce0f55-034a-4127-82e2-52744178bc32.jpg)

Option (B). If you install mmpose with pip, open you python interpreter and copy & paste the following codes.

```python
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# please prepare an image with person
results = inference_topdown(model, 'demo.jpg')
```

The `demo.jpg` can be downloaded from [Github](https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg).

The inference results will be a list of `PoseDataSample`, and the predictions are in the `pred_instances`, indicating the detected keypoint locations and scores.

## Customize Installation

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.

### Install MMEngine without MIM

To install MMEngine with pip instead of MIM, please follow [MMEngine installation guides](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html).

For example, you can install MMEngine by the following command.

```shell
pip install mmengine
```

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install 'mmcv>=2.0.1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Install on CPU-only platforms

MMPose can be built for CPU only environment. In CPU mode you can train, test or inference a model.

However, some functionalities are missing in this mode, usually GPU-compiled ops like `Deformable Convolution`. Most models in MMPose don't depend on these ops, but if you try to train/test/infer a model containing these ops, an error will be raised.

### Install on Google Colab

[Google Colab](https://colab.research.google.com/) usually has PyTorch installed,
thus we only need to install MMEngine, MMCV and MMPose with the following commands.

**Step 1.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.1"
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
# Example output: 1.1.0
```

```{note}
Note that within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

### Using MMPose with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmpose/blob/master/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.8.0, CUDA 10.1, CUDNN 7.
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

If you have some issues during the installation, please first view the [FAQ](./faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmpose/issues/new/choose) on GitHub if no solution is found.
