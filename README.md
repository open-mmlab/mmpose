# 多摄像头人体姿态估计系统

## 项目简介

这是一个基于MMPose的多摄像头人体姿态估计系统，能够实时捕获和分析多个摄像头中的人体姿态，支持人员跟踪和骨架可视化。

## 功能特点

- 支持多摄像头并行处理
- 实时人体姿态估计
- 人员ID跟踪与标记
- 高性能GPU加速处理
- 骨架和关键点可视化
- 实时FPS和推理时间统计

## 系统要求

- Python 3.7+
- CUDA 9.2+ (推荐CUDA 11.0+)
- PyTorch 1.8+
- 至少两个摄像头设备
- 操作系统：Windows/Linux/macOS

## 安装教程

### 先决条件

**步骤 0.** 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

**步骤 1.** 创建并激活conda环境。

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**步骤 2.** 安装PyTorch（支持CUDA）

对于GPU平台：
```bash
conda install pytorch torchvision -c pytorch
```
或者指定CUDA版本：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

对于仅CPU平台：
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

### 安装依赖库

**步骤 3.** 使用MIM安装MMEngine和MMCV。

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```

**步骤 4.** 安装MMPose（两种方式）

方式一：使用pip安装MMPose包（推荐一般用户）
```bash
mim install "mmpose>=1.1.0"
```

方式二：从源代码安装（推荐开发者）
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

**步骤 5.** 安装其他依赖

```bash
pip install opencv-python numpy
```

**步骤 6.** （可选）安装MMDetection用于人体检测

如果需要运行一些依赖MMDetection的演示脚本，可以安装MMDetection：

```bash
mim install "mmdet>=3.1.0"
```

### 验证安装

验证MMPose是否正确安装：

```bash
# 下载配置和检查点文件
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest .

# 运行推理演示
python -c "
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  # 或 device='cpu'

# 请准备一张有人的图片
results = inference_topdown(model, '测试图片.jpg')
print('推理成功!')
"
```

### 其他安装选项

#### 在Google Colab上安装

```bash
!pip3 install openmim
!mim install mmengine
!mim install "mmcv>=2.0.1"
!git clone https://github.com/open-mmlab/mmpose.git
%cd mmpose
!pip install -e .
```

#### 使用Docker

```bash
# 构建镜像
docker build -t mmpose docker/

# 运行容器
docker run --gpus all --shm-size=8g -it -v {数据目录}:/mmpose/data mmpose
```

## 使用方法

1. 连接至少两个摄像头到计算机
2. 运行程序:

```bash
python multi_camera_pose_estimation/multi_camera.py
```

3. 按'q'键退出程序

## 自定义配置

可以在`main()`函数中修改以下参数：

- 模型选择：可选择`rtmpose-l_8xb32-270e_coco-wholebody-384x288`或其他MMPose支持的模型
- 摄像头ID：默认使用ID为0和1的两个摄像头
- 关键点阈值：可调整`kpt_thr`参数以控制关键点检测的灵敏度
- 边界框阈值：可调整`bbox_thr`参数以控制人体检测的灵敏度

## 模型库(ModelZoo)

本项目使用MMPose提供的全身人体姿态估计(Wholebody 2D Keypoint)模型。以下是可用的主要模型系列及其性能特点：

### 1. RTMPose系列模型

RTMPose是一系列实时多人姿态估计模型，针对实时应用场景进行了优化，提供了不同规模和精度的变体：

#### 在COCO-Wholebody数据集上的性能表现

| 模型 | 输入尺寸 | 躯干AP | 足部AP | 面部AP | 手部AP | 全身AP | 模型文件 |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|:-------:|
| rtmpose-m | 256x192 | 0.680 | 0.619 | 0.842 | 0.516 | 0.606 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth) |
| rtmpose-l | 256x192 | 0.704 | 0.672 | 0.876 | 0.536 | 0.635 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth) |
| rtmpose-l | 384x288 | 0.712 | 0.693 | 0.882 | 0.579 | 0.648 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth) |

### 2. RTMW系列模型

RTMW是基于RTMPose的全身姿态估计高级模型，在Cocktail14多数据集上训练，具有更好的泛化能力：

#### 在Cocktail14数据集上的性能表现

| 模型 | 输入尺寸 | 躯干AP | 足部AP | 面部AP | 手部AP | 全身AP | 模型文件 |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|:-------:|
| rtmw-m | 256x192 | 0.676 | 0.671 | 0.783 | 0.491 | 0.582 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth) |
| rtmw-l | 256x192 | 0.743 | 0.763 | 0.834 | 0.598 | 0.660 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth) |
| rtmw-x | 256x192 | 0.746 | 0.770 | 0.844 | 0.610 | 0.672 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth) |
| rtmw-l | 384x288 | 0.761 | 0.793 | 0.884 | 0.663 | 0.701 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth) |
| rtmw-x | 384x288 | 0.763 | 0.796 | 0.884 | 0.664 | 0.702 | [下载链接](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth) |

### 3. HRNet系列模型

HRNet是经典的高分辨率网络架构，适用于高精度姿态估计任务：

#### 在UBody-COCO-Wholebody数据集上的性能表现

| 模型 | 输入尺寸 | 躯干AP | 足部AP | 面部AP | 手部AP | 全身AP | 模型文件 |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|:-------:|
| pose_hrnet_w32 | 256x192 | 0.685 | 0.564 | 0.625 | 0.516 | 0.549 | [下载链接](https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/ubody/td-hm_hrnet-w32_8xb64-210e_ubody-coco-256x192-7c227391_20230807.pth) |

### 如何在本项目中使用这些模型

1. 下载所需模型权重文件

```bash
# 示例：下载RTMPose-L模型
mim download mmpose --config rtmpose-l_8xb32-270e_coco-wholebody-384x288 --dest .
```

2. 在`main()`函数中修改模型配置

```python
# 使用RTMPose-L模型示例
process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, 'rtmpose-l_8xb32-270e_coco-wholebody-384x288', device))
process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, 'rtmpose-l_8xb32-270e_coco-wholebody-384x288', device))

# 使用RTMW-X模型示例（更高精度，但可能更慢）
# process1 = mp.Process(target=camera_process, args=(0, return_dict, shared_data, 'rtmw-x_8xb320-270e_cocktail14-384x288', device))
# process2 = mp.Process(target=camera_process, args=(1, return_dict, shared_data, 'rtmw-x_8xb320-270e_cocktail14-384x288', device))
```

3. 模型选择建议
   - 对于需要高帧率的实时应用，推荐使用RTMPose-M或RTMPose-L模型
   - 对于需要更高精度的应用，推荐使用RTMW-L或RTMW-X模型
   - 输入尺寸较大的模型(384x288)精度更高，但推理速度稍慢

## 性能优化

- 使用多进程并行处理多个摄像头输入
- CUDA加速和半精度推理(FP16)优化
- 内存复用以减少内存分配开销
- 图像编码质量优化以加快帧传输

## 输出示例

程序运行后会显示两个窗口，分别展示两个摄像头的实时姿态估计结果，包括：

- 检测到的人体骨架
- 每个人的唯一ID标识
- 实时FPS和推理时间统计

## 注意事项

- 确保摄像头工作正常且被系统识别
- 推荐使用高性能GPU以获得最佳实时性能
- 在光线良好的环境中使用，可提高检测准确率
- CUDA版本需要与PyTorch版本匹配，否则可能出现`No module named 'mmcv.ops'`或`No module named 'mmcv._ext'`错误

## 参考资料

- [MMPose官方安装文档](https://mmpose.readthedocs.io/en/latest/installation.html)
- [MMPose全身姿态估计模型库](https://mmpose.readthedocs.io/en/latest/model_zoo/wholebody_2d_keypoint.html)

