# 安装

我们推荐用户按照我们的最佳实践来安装 MMPose。但除此之外，如果您想根据
您的习惯完成安装流程，也可以参见 [自定义安装](#自定义安装) 一节来获取更多信息。

- [安装](#安装)
  - [依赖环境](#依赖环境)
  - [最佳实践](#最佳实践)
    - [从源码安装 MMPose](#从源码安装-mmpose)
    - [作为 Python 包安装](#作为-python-包安装)
  - [验证安装](#验证安装)
  - [自定义安装](#自定义安装)
    - [CUDA 版本](#cuda-版本)
    - [不使用 MIM 安装 MMEngine](#不使用-mim-安装-mmengine)
    - [在 CPU 环境中安装](#在-cpu-环境中安装)
    - [在 Google Colab 中安装](#在-google-colab-中安装)
    - [通过 Docker 使用 MMPose](#通过-docker-使用-mmpose)
  - [故障解决](#故障解决)

## 依赖环境

在本节中，我们将演示如何准备 PyTorch 相关的依赖环境。

MMPose 适用于 Linux、Windows 和 macOS。它需要 Python 3.7+、CUDA 9.2+ 和 PyTorch 1.8+。

如果您对配置 PyTorch 环境已经很熟悉，并且已经完成了配置，可以直接进入下一节：[安装](#安装-mmpose)。否则，请依照以下步骤完成配置。

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

**第 4 步** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x)

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```

请注意，MMPose 中的一些推理示例脚本需要使用 [MMDetection](https://github.com/open-mmlab/mmdetection) (mmdet) 检测人体。如果您想运行这些示例脚本，可以通过运行以下命令安装 mmdet:

```shell
mim install "mmdet>=3.1.0"
```

## 最佳实践

根据具体需求，我们支持两种安装模式: 从源码安装（推荐）和作为 Python 包安装

### 从源码安装（推荐）

如果基于 MMPose 框架开发自己的任务，需要添加新的功能，比如新的模型或是数据集，或者使用我们提供的各种工具。从源码按如下方式安装 mmpose：

```shell
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" 表示输出更多安装相关的信息
# "-e" 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效
```

### 作为 Python 包安装

如果只是希望调用 MMPose 的接口，或者在自己的项目中导入 MMPose 中的模块。直接使用 mim 安装即可。

```shell
mim install "mmpose>=1.1.0"
```

## 验证安装

为了验证 MMPose 是否安装正确，您可以通过以下步骤运行模型推理。

**第 1 步** 我们需要下载配置文件和模型权重文件

```shell
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .
```

下载过程往往需要几秒或更多的时间，这取决于您的网络环境。完成之后，您会在当前目录下找到这两个文件：`td-hm_hrnet-w48_8xb32-210e_coco-256x192.py` 和 `hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth`, 分别是配置文件和对应的模型权重文件。

**第 2 步** 验证推理示例

如果您是**从源码安装**的 mmpose，可以直接运行以下命令进行验证：

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --out-file vis_results.jpg \
    --draw-heatmap
```

如果一切顺利，您将会得到这样的可视化结果：

![image](https://user-images.githubusercontent.com/87690686/187824033-2cce0f55-034a-4127-82e2-52744178bc32.jpg)

代码会将预测的关键点和热图绘制在图像中的人体上，并保存到当前文件夹下的 `vis_results.jpg`。

如果您是**作为 Python 包安装**，可以打开您的 Python 解释器，复制并粘贴如下代码：

```python
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# 请准备好一张带有人体的图片
results = inference_topdown(model, 'demo.jpg')
```

示例图片 `demo.jpg` 可以从 [Github](https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/coco/000000000785.jpg) 下载。
推理结果是一个 `PoseDataSample` 列表，预测结果将会保存在 `pred_instances` 中，包括检测到的关键点位置和置信度。

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

### 不使用 MIM 安装 MMEngine

若不使用 mim 安装 MMEngine，请遵循 [ MMEngine 安装指南](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html).

例如，您可以通过以下命令安装 MMEngine:

```shell
pip install mmengine
```

### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此其对 PyTorch 的依赖比较复杂。MIM 会自动解析这些
依赖，选择合适的 MMCV 预编译包，使安装更简单，但它并不是必需的。

若不使用 mim 来安装 MMCV，请遵照 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)。
它需要您用指定 url 的形式手动指定对应的 PyTorch 和 CUDA 版本。

举个例子，如下命令将会安装基于 PyTorch 1.10.x 和 CUDA 11.3 编译的 mmcv。

```shell
pip install 'mmcv>=2.0.1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 在 CPU 环境中安装

MMPose 可以仅在 CPU 环境中安装，在 CPU 模式下，您可以完成训练、测试和模型推理等所有操作。

在 CPU 模式下，MMCV 的部分功能将不可用，通常是一些 GPU 编译的算子，如 `Deformable Convolution`。MMPose 中大部分的模型都不会依赖这些算子，但是如果您尝试使用包含这些算子的模型来运行训练、测试或推理，将会报错。

### 在 Google Colab 中安装

[Google Colab](https://colab.research.google.com/) 通常已经包含了 PyTorch 环境，因此我们只需要安装 MMEngine, MMCV 和 MMPose 即可，命令如下：

**第 1 步** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x)

```shell
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.1"
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
# 预期输出： 1.1.0
```

```{note}
在 Jupyter 中，感叹号 `!` 用于执行外部命令，而 `%cd` 是一个[魔术命令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd)，用于切换 Python 的工作路径。
```

### 通过 Docker 使用 MMPose

MMPose 提供 [Dockerfile](https://github.com/open-mmlab/mmpose/blob/master/docker/Dockerfile)
用于构建镜像。请确保您的 [Docker 版本](https://docs.docker.com/engine/install/) >=19.03。

```shell
# 构建默认的 PyTorch 1.8.0，CUDA 10.1 版本镜像
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
