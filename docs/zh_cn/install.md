# 安装

本文档提供了安装 MMPose 的相关步骤。

<!-- TOC -->

- [安装依赖包](#安装依赖包)
- [准备环境](#准备环境)
- [MMPose 的安装步骤](#MMPose-的安装步骤)
- [CPU 环境下的安装步骤](#CPU-环境下的安装步骤)
- [利用 Docker 镜像安装 MMPose](#利用-Docker-镜像安装-MMPose)
- [源码安装 MMPose](#源码安装-MMPose)
- [在多个 MMPose 版本下进行开发](#在多个-MMPose-版本下进行开发)

<!-- TOC -->

## 安装依赖包

- Linux | Windows | macOS
- Python 3.6+
- PyTorch 1.5+
- CUDA 9.2+ (如果从源码编译 PyTorch,则可以兼容 CUDA 9.0 版本)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) 请安装最新版本的 mmcv-full
- Numpy
- cv2
- json_tricks
- [xtcocotools](https://github.com/jin-s13/xtcocoapi)

可选项：

- [mmdet](https://github.com/open-mmlab/mmdetection) (用于“姿态估计”)
- [mmtrack](https://github.com/open-mmlab/mmtracking) (用于“姿态跟踪”)
- [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html) (用于“三维人体形状恢复”)
- [smplx](https://github.com/vchoutas/smplx) (用于“三维人体形状恢复”)

## 准备环境

a. 创建并激活 conda 虚拟环境，如：

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. 参考 [官方文档](https://pytorch.org/) 安装 PyTorch 和 torchvision ，如：

```shell
conda install pytorch torchvision -c pytorch
```

**注**：确保 CUDA 的编译版本和 CUDA 的运行版本相匹配。
用户可以参照 [PyTorch 官网](https://pytorch.org/) 对预编译包所支持的 CUDA 版本进行核对。

`例 1`：如果用户的 `/usr/local/cuda` 文件夹下已安装 CUDA 10.2 版本，并且想要安装 PyTorch 1.8.0 版本，
则需要安装 CUDA 10.2 下预编译的 PyTorch。

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

`例 2`：如果用户的 `/usr/local/cuda` 文件夹下已安装 CUDA 9.2 版本，并且想要安装 PyTorch 1.7.0 版本，
则需要安装 CUDA 9.2 下预编译的 PyTorch。

```shell
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=9.2 -c pytorch
```

如果 PyTorch 是由源码进行编译安装（而非直接下载预编译好的安装包），则可以使用更多的 CUDA 版本（如 9.0 版本）。

## MMPose 的安装步骤

a. 安装最新版本的 mmcv-full。MMPose 推荐用户使用如下的命令安装预编译好的 mmcv。

```shell
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
# 我们可以忽略 PyTorch 的小版本号
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9/index.html
```

PyTorch 在 1.x.0 和 1.x.1 之间通常是兼容的，故 mmcv-full 只提供 1.x.0 的编译包。如果你的 PyTorch 版本是 1.x.1，你可以放心地安装在 1.x.0 版本编译的 mmcv-full。

可查阅 [这里](https://github.com/open-mmlab/mmcv#installation) 以参考不同版本的 MMCV 所兼容的 PyTorch 和 CUDA 版本。

另外，用户也可以通过使用以下命令从源码进行编译：

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # mmcv-full 包含一些 cuda 算子，执行该步骤会安装 mmcv-full（而非 mmcv）
# 或者使用 pip install -e .  # 这个命令安装的 mmcv 将不包含 cuda ops，通常适配 CPU（无 GPU）环境
cd ..
```

**注意**：如果之前安装过 mmcv，那么需要先使用 `pip uninstall mmcv` 命令进行卸载。如果 mmcv 和 mmcv-full 同时被安装, 会报 `ModuleNotFoundError` 的错误。

b. 克隆 MMPose 库。

```shell
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
```

c. 安装依赖包和 MMPose。

```shell
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
```

如果是在 macOS 环境安装 MMPose，则需使用如下命令：

```shell
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

d. 安装其他可选依赖。

如果用户不需要做相关任务，这部分步骤可以选择跳过。

可选项：

- [mmdet](https://github.com/open-mmlab/mmdetection) (用于“姿态估计”)
- [mmtrack](https://github.com/open-mmlab/mmtracking) (用于“姿态跟踪”)
- [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html) (用于“三维人体形状恢复”)
- [smplx](https://github.com/vchoutas/smplx) (用于“三维人体形状恢复”)

注意：

1. 在步骤 c 中，git commit 的 id 将会被写到版本号中，如 0.6.0+2e7045c。这个版本号也会被保存到训练好的模型中。
   这里推荐用户每次在步骤 b 中对本地代码和 github 上的源码进行同步。如果 C++/CUDA 代码被修改，就必须进行这一步骤。

1. 根据上述步骤，MMPose 就会以 `dev` 模式被安装，任何本地的代码修改都会立刻生效，不需要再重新安装一遍（除非用户提交了 commits，并且想更新版本号）。

1. 如果用户想使用 `opencv-python-headless` 而不是 `opencv-python`，可再安装 MMCV 前安装 `opencv-python-headless`。

1. 如果 mmcv 已经被安装，用户需要使用 `pip uninstall mmcv` 命令进行卸载。如果 mmcv 和 mmcv-full 同时被安装, 会报 `ModuleNotFoundError` 的错误。

1. 一些依赖包是可选的。运行 `python setup.py develop` 将只会安装运行代码所需的最小要求依赖包。
   要想使用一些可选的依赖包，如 `smplx`，用户需要通过 `pip install -r requirements/optional.txt` 进行安装，
   或者通过调用 `pip`（如 `pip install -v -e .[optional]`，这里的 `[optional]` 可替换为 `all`，`tests`，`build` 或 `optional`） 指定安装对应的依赖包，如 `pip install -v -e .[tests,build]`。

## CPU 环境下的安装步骤

MMPose 可以在只有 CPU 的环境下安装（即无法使用 GPU 的环境）。

在 CPU 模式下，用户可以运行 `demo/demo.py` 的代码。

## 源码安装 MMPose

这里提供了 conda 下安装 MMPose 并链接 COCO 数据集路径的完整脚本（假设 COCO 数据的路径在 $COCO_ROOT）。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# 安装最新的，使用默认版本的 CUDA 版本（一般为最新版本）预编译的 PyTorch 包
conda install -c pytorch pytorch torchvision -y

# 安装 mmcv-full。其中，命令里 url 的 ``{cu_version}`` 和 ``{torch_version}`` 变量需由用户进行指定。
# 可查阅 [这里](https://github.com/open-mmlab/mmcv#installation) 以参考不同版本的 MMCV 所兼容的 PyTorch 和 CUDA 版本。
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

# 安装 mmpose
git clone git@github.com:open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
python setup.py develop

mkdir data
ln -s $COCO_ROOT data/coco
```

## 利用 Docker 镜像安装 MMPose

MMPose 提供一个 [Dockerfile](/docker/Dockerfile) 用户创建 docker 镜像。

```shell
# 创建拥有 PyTorch 1.6.0, CUDA 10.1, CUDNN 7 配置的 docker 镜像.
docker build -f ./docker/Dockerfile --rm -t mmpose .
```

**注意**：用户需要确保已经安装了 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。

运行以下命令：

```shell
docker run --gpus all\
 --shm-size=8g \
 -it -v {DATA_DIR}:/mmpose/data mmpose
```

## 在多个 MMPose 版本下进行开发

MMPose 的训练和测试脚本已经修改了 `PYTHONPATH` 变量，以确保其能够运行当前目录下的 MMPose。

如果想要运行环境下默认的 MMPose，用户需要在训练和测试脚本中去除这一行：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
