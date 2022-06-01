<!-- TOC -->

- [依赖环境](#依赖环境)
- [安装](#安装)
  - [最佳实践](#最佳实践)
    - [从源码安装](#从源码安装)
    - [作为 Python 包安装](#作为-python-包安装)
  - [验证安装](#验证安装)
  - [自定义安装](#自定义安装)
    - [CUDA 版本](#cuda-版本)
    - [不使用 MIM 安装 MMCV](#不使用-mim-安装-mmcv)
    - [在 CPU 环境中安装](#在-cpu-环境中安装)
    - [在 Google Colab 中安装](#在-google-colab-中安装)
    - [通过 Docker 使用 MMPose](#通过-docker-使用-mmpose)
  - [故障解决](#故障解决)

<!-- TOC -->

# 依赖环境

在本节中，我们将演示如何准备 PyTorch 相关的依赖环境。

MMPose 适用于 Linux、Windows 和 macOS。它需要 Python 3.6+、CUDA 9.2+ 和 PyTorch 1.5+。

```{note}
如果您对配置 PyTorch 环境已经很熟悉，并且已经完成了配置，可以直接进入[下一节](#安装)。
否则，请依照以下步骤完成配置。
```

**第 1 步** 从[官网](https://docs.conda.io/en/latest/miniconda.html) 下载并安装 Miniconda。

**第 2 步** 创建一个 conda 虚拟环境并激活它。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**第 3 步** 按照[官方指南](https://pytorch.org/get-started/locally/) 安装 PyTorch。例如：

在 GPU 平台：

```shell
conda install pytorch torchvision -c pytorch
```

```{warning}
以上命令会自动安装最新版的 PyTorch 与对应的 cudatoolkit，请检查它们是否与您的环境匹配。
```

在 CPU 平台：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# 安装

我们推荐用户按照我们的最佳实践来安装 MMPose。但除此之外，如果您想根据
您的习惯完成安装流程，也可以参见[自定义安装](#自定义安装)一节来获取更多信息。

## 最佳实践

**第 1 步** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
pip install -U openmim
mim install mmcv-full
```

**第 2 步** 安装 MMPose

根据具体需求，我们支持两种安装模式：

- [从源码安装（推荐）](#从源码安装)：如果基于 MMPose 框架开发自己的任务，需要添加新的功能，比如新的模型或是数据集，或者使用我们提供的各种工具。
- [作为 Python 包安装](#作为-python-包安装)：只是希望调用 MMPose 的接口，或者在自己的项目中导入 MMPose 中的模块。

### 从源码安装

这种情况下，从源码按如下方式安装 mmpose：

```shell
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" 表示输出更多安装相关的信息
# "-e" 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效
```

### 作为 Python 包安装

直接使用 pip 安装即可。

```shell
pip install mmpose
```

## 验证安装

为了验证 MMPose 的安装是否正确，我们提供了一些示例代码来执行模型推理。

**第 1 步** 我们需要下载配置文件和模型权重文件

```shell
mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest .
```

下载过程往往需要几秒或更多的时间，这取决于您的网络环境。完成之后，您会在当前目录下找到这两个文件：`associative_embedding_hrnet_w32_coco_512x512.py`, `hrnet_w32_coco_512x512-bcb8c247_20200816.pth`, 分别是配置文件和对应的模型权重文件。

**第 2 步** 验证推理示例

如果您是**从源码安装**的 mmpose，那么直接运行以下命令进行验证：

```shell
python demo/bottom_up_img_demo.py associative_embedding_hrnet_w32_coco_512x512.py hrnet_w32_coco_512x512-bcb8c247_20200816.pth --img-path tests/data/coco/ --out-img-root vis_results
```

您可以在 `vis_results` 这个目录下看到输出的图片，这些图片展示了人体姿态估计的结果。

如果您是**作为 PyThon 包安装**，那么可以打开您的 Python 解释器，复制并粘贴如下代码：

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

准备好一张带有人的图片，并放置在合适的位置，然后运行以上代码，您将会在输出的图片上看到检测到的人体姿态结果。

## 自定义安装

### CUDA 版本

安装 PyTorch 时，需要指定 CUDA 版本。如果您不清楚选择哪个，请遵循我们的建议：

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 系列 以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向后兼容 (backward compatible) 的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保您的 GPU 驱动版本满足最低的版本需求，参阅[这张表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果按照我们的最佳实践进行安装，CUDA 运行时库就足够了，因为我们提供相关 CUDA 代码的预编译，您不需要进行本地编译。
但如果您希望从源码进行 MMCV 的编译，或是进行其他 CUDA 算子的开发，那么就必须安装完整的 CUDA 工具链，参见
[NVIDIA 官网](https://developer.nvidia.com/cuda-downloads)，另外还需要确保该 CUDA 工具链的版本与 PyTorch 安装时
的配置相匹配（如用 `conda install` 安装 PyTorch 时指定的 cudatoolkit 版本）。
```

### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此其对 PyTorch 的依赖比较复杂。MIM 会自动解析这些
依赖，选择合适的 MMCV 预编译包，使安装更简单，但它并不是必需的。

要使用 pip 而不是 MIM 来安装 MMCV，请遵照 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html)。
它需要您用指定 url 的形式手动指定对应的 PyTorch 和 CUDA 版本。

举个例子，如下命令将会安装基于 PyTorch 1.10.x 和 CUDA 11.3 编译的 mmcv-full。

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 在 CPU 环境中安装

MMPose 可以仅在 CPU 环境中安装，在 CPU 模式下，您可以完成训练（需要 MMCV 版本 >= 1.4.4）、测试和模型推理等所有操作。

在 CPU 模式下，MMCV 的部分功能将不可用，通常是一些 GPU 编译的算子，如 `Deformable Convolution`。MMPose 中大部分的模型都不会依赖这些算子，但是如果您尝试使用包含这些算子的模型来运行训练、测试或推理，将会报错。

### 在 Google Colab 中安装

[Google Colab](https://colab.research.google.com/) 通常已经包含了 PyTorch 环境，因此我们只需要安装 MMCV 和 MMPose 即可，命令如下：

**第 1 步** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
!pip3 install openmim
!mim install mmcv-full
```

**第 2 步** 从源码安装 mmpose

```shell
!git clone https://github.com/open-mmlab/mmpose.git
%cd mmpose
!pip install -e .
```

**第 3 步** 验证

```python
import mmpose
print(mmpose.__version__)
# 预期输出： 0.26.0 或其他版本号
```

```{note}
在 Jupyter 中，感叹号 `!` 用于执行外部命令，而 `%cd` 是一个[魔术命令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd)，用于切换 Python 的工作路径。
```

### 通过 Docker 使用 MMPose

MMPose 提供 [Dockerfile](https://github.com/open-mmlab/mmpose/blob/master/docker/Dockerfile)
用于构建镜像。请确保您的 [Docker 版本](https://docs.docker.com/engine/install/) >=19.03。

```shell
# 构建默认的 PyTorch 1.6.0，CUDA 10.1 版本镜像
# 如果您希望使用其他版本，请修改 Dockerfile
docker build -t mmpose docker/
```

**注意**：请确保您已经安装了 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。

用以下命令运行 Docker 镜像：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmpose/data mmpose
```

`{DATA_DIR}` 是您本地存放用于 MMPose 训练、测试、推理等流程的数据目录。

## 故障解决

如果您在安装过程中遇到了什么问题，请先查阅[常见问题](faq.md)。如果没有找到解决方法，可以在 GitHub
上[提出 issue](https://github.com/open-mmlab/mmpose/issues/new/choose)。
