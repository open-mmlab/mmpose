# Installation

<!-- TOC -->

- [Requirements](#requirements)
- [Prepare Environment](#prepare-environment)
- [Install MMPose](#install-mmpose)
- [Install with CPU only](#install-with-cpu-only)
- [A from-scratch setup script](#a-from-scratch-setup-script)
- [Another option: Docker Image](#another-option-docker-image)
- [Developing with multiple MMPose versions](#developing-with-multiple-mmpose-versions)

<!-- TOC -->

## Requirements

- Linux | Windows | macOS
- Python 3.6+
- PyTorch 1.5+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) (Please install the latest version of mmcv-full)
- Numpy
- cv2
- json_tricks
- [xtcocotools](https://github.com/jin-s13/xtcocoapi)

Optional:

- [mmdet](https://github.com/open-mmlab/mmdetection) (to run pose demos)
- [mmtrack](https://github.com/open-mmlab/mmtracking) (to run pose tracking demos)
- [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html) (to run 3d mesh demos)
- [smplx](https://github.com/vchoutas/smplx) (to run 3d mesh demos)

## Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

```{note}
Make sure that your compilation CUDA version and runtime CUDA version match.
```

You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.8.0,
you need to install the prebuilt PyTorch with CUDA 10.2.

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

`E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.7.0.,
you need to install the prebuilt PyTorch with CUDA 9.2.

```shell
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=9.2 -c pytorch
```

If you build PyTorch from source instead of installing the pre-built package, you can use more CUDA versions such as 9.0.

## Install MMPose

a. Install mmcv, we recommend you to install the pre-built mmcv as below.

```shell
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
# We can ignore the micro version of PyTorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9/index.html
```

mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

**Important:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

b. Clone the mmpose repository

```shell
git clone git@github.com:open-mmlab/mmpose.git # or git clone https://github.com/open-mmlab/mmpose
cd mmpose
```

c. Install build requirements and then install mmpose

```shell
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build MMPose on macOS, replace the last command with

```shell
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

d. Install optional modules

- [mmdet](https://github.com/open-mmlab/mmdetection) (to run pose demos)
- [mmtrack](https://github.com/open-mmlab/mmtracking) (to run pose tracking demos)
- [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html) (to run 3d mesh demos)
- [smplx](https://github.com/vchoutas/smplx) (to run 3d mesh demos)

```{note}
1. The git commit id will be written to the version number with step c, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
   It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

1. Following the above instructions, mmpose is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

1. If you would like to use `opencv-python-headless` instead of `opencv-python`,
   you can install it before installing MMCV.

1. If you have `mmcv` installed, you need to firstly uninstall `mmcv`, and then install `mmcv-full`.

1. Some dependencies are optional. Running `python setup.py develop` will only install the minimum runtime requirements.
   To use optional dependencies like `smplx`, either install them with `pip install -r requirements/optional.txt`
   or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`,
   valid keys for the `[optional]` field are `all`, `tests`, `build`, and `optional`) like `pip install -v -e .[tests,build]`.
```

## Install with CPU only

The code can be built for CPU only environment (where CUDA isn't available).

In CPU mode you can run the demo/demo.py for example.

## A from-scratch setup script

Here is a full script for setting up mmpose with conda and link the dataset path (supposing that your COCO dataset path is $COCO_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install the latest mmcv-full
# Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one.
# See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

# install mmpose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .

mkdir data
ln -s $COCO_ROOT data/coco
```

## Another option: Docker Image

We provide a [Dockerfile](/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6.0, CUDA 10.1, CUDNN 7.
docker build -f ./docker/Dockerfile --rm -t mmpose .
```

**Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Run the following cmd:

```shell
docker run --gpus all\
 --shm-size=8g \
 -it -v {DATA_DIR}:/mmpose/data mmpose
```

## Developing with multiple MMPose versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMPose in the current directory.

To use the default MMPose installed in the environment rather than that you are working with, you can remove the following line in those scripts.

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
