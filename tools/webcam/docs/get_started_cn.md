# MMPose Webcam API 快速上手

## 什么是 MMPose Webcam API

MMPose WebcamAPI 是一套简单的应用开发接口，可以帮助用户方便的调用 MMPose 以及其他 OpenMMLab 算法库中的算法，实现基于摄像头输入视频的交互式应用。

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/153800450-2522efe8-bc11-457d-9037-d8aee4fc4f36.png">
</div>
MMPose Webcam API 框架概览
<div align=center>
</div>

## 运行一个 Demo

我们将从一个简单的 Demo 开始，向您介绍 MMPose WebcamAPI 的功能和特性，并详细展示如何基于这个 API 搭建自己的应用。为了使用 MMPose WebcamAPI，您只需要做简单的准备：

1. 一台计算机（最好有 GPU 和 CUDA 环境，但这并不是必须的）
1. 一个摄像头。计算机自带摄像头或者外接 USB 摄像头均可
1. 安装 MMPose
    - 在 OpenMMLab [官方仓库](https://github.com/open-mmlab/mmpose) fork MMPose 到自己的 github，并 clone 到本地
    - 安装 MMPose，只需要按照我们的 [安装文档](https://mmpose.readthedocs.io/zh_CN/latest/install.html) 中的步骤操作即可

完成准备工作后，请在命令行进入 MMPose 根目录，执行以下指令，即可运行 demo：

```shell
python tools/webcam/run_webcam.py --config tools/webcam/configs/examples/pose_estimation.py
```

这个 demo 实现了目标检测，姿态估计和可视化功能，效果如下：

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/153772158-a5702193-3d3f-40c8-bd6b-ab186979d1b4.png">
</div>
Pose Estimation Demo 效果
<div align=center>
</div>

## Demo 里面有什么？

### 从 Config 说起

成功运行 demo 后，我们来看一下它是怎样工作的。在启动脚本 `tools/webcam/run_webcam.py` 中可以看到，这里的操作很简单：首先读取了一个 config 文件，接着使用 config 构建了一个 runner ，最后调用了 runner 的 `run()` 方法，这样 demo 就开始运行了。

```python
# tools/webcam/run_webcam.py

def launch():
    # 读取 config 文件
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # 构建 runner（WebcamRunner类的实例）
    runner = WebcamRunner(**cfg.runner)
    # 调用 run()方法，启动程序
    runner.run()


if __name__ == '__main__':
    launch()
```

我们先不深究 runner 为何物，而是接着看一下这个 config 文件的内容。省略掉细节和注释，可以发现 config 的结构大致包含两部分（如下图所示）：

1. Runner 的基本参数，如 camera_id，camera_fps 等。这部分比較好理解，是一些在读取视频时的必要设置
2. 一系列＂节点＂（Node），每个节点属于特定的类型（type），并有对应的一些参数

```python
runner = dict(
    # runner的基本参数
    name='Pose Estimation',
    camera_id=0,
    camera_fps=20,
    # 定义了若干节点(Node)
    Nodes=[
        dict(
            type='DetectorNode',  # 节点１类型
            name='Detector',  # 节点１名字
            input_buffer='_input_',  # 节点１数据输入
            output_buffer='det_result',  # 节点１数据输出
            ...), # 节点１其他参数
        dict(
            type='TopDownPoseEstimatorNode',  # 节点２类型
            name='Human Pose Estimator',  # 节点２名字
            input_buffer='det_result',  # 节点2数据输入
            output_buffer='pose_result',  # 节点2数据输出
            ...),  # 节点２参数
        ...]  # 更多节点
)
```

### 核心概念：Runner 和 Node

到这里，我们已经引出了 MMPose WebcamAPI 的２个最重要的概念：runner 和 Node，下面做正式介绍：

- Runner：Runner 类是程序的主体，提供了程序启动的入口runner.run()方法，并负责视频读入，输出显示等功能。此外，runner 中会包含若干个 Node，分别负责在视频帧的处理中执行不同的功能。
- Node：Node 类用来定义功能模块，例如模型推理，可视化，特效绘制等都可以通过定义一个对应的 Node 来实现。如上面的 config 例子中，2 个节点的功能分别是做目标检测（Detector）和姿态估计（TopDownPoseEstimator）

Runner 和 Node 的关系简单来说如下图所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/153772839-104430bd-de0e-4ee5-bd67-dff4e52d784c.png">
</div>
Runner 和 Node 逻辑关系示意
<div align=center>
</div>

### 数据流

一个重要的问题是：当一帧视频数据被 runner 读取后，会按照怎样的顺序通过所有的 Node 并最终被输出（显示）呢？
答案就是 config 中每个 Node 的输入输出配置。如示例 config 中，可以看到每个 Node 都有`input_buffer`，`output_buffer`等参数，用来定义该节点的输入输出。通过这种连接关系，所有的 Node 构成了一个有向无环图结构，如下图所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/153772900-9619ae80-3e64-4b40-bc1e-b184405e3d5b.png">
</div>
数据流示意
<div align=center>
</div>

图中的每个 Data Buffer 就是一个用来存放数据的容器。用户不需要关注 buffer 的具体细节，只需要将其简单理解成 Node 输入输出的名字即可。用户在 config 中可以任意定义这些名字，不过要注意有以下几个特殊的名字：

- \_input\_：存放 runner 读入的视频帧，用于模型推理
- \_frame\_ ：存放 runner 读入的视频帧，用于可视化
- \_display\_：存放经过所以 Node 处理后的结果，用于在屏幕上显示

当一帧视频数据被 runner 读入后，会被放进 _input_ 和 _frame_ 两个 buffer 中，然后按照 config 中定义的 Node 连接关系依次通过各个 Node ，最终到达 _display_ ，并被 runner 读出显示在屏幕上。

#### Get Advanced: 关于 buffer

- Buffer 本质是一个有限长度的队列，在 runner 中会包含一个 BufferManager 实例（见`mmpose/tools/webcam/webcam_apis/buffer.py'）来生成和管理所有 buffer。Node 会按照 config 从对应的 buffer 中读出或写入数据。
- 当一个 buffer 已满（达到最大长度）时，写入数据的操作通常不会被 block，而是会将 buffer 中已有的最早一条数据“挤出去”。
- 为什么有_input_和_frame_两个输入呢？因为有些 Node 的操作较为耗时（如目标检测，姿态估计等需要模型推理的 Node）。为了保证显示的流畅，我们通常用_input_来作为这类耗时较大的操作的输入，而用_frame_来实时绘制可视化的结果。因为各个节点是异步运行的，这样就可以保证可视化的实时和流畅。
