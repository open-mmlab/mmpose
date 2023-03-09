# 教程 7：摄像头应用接口

MMPose 中提供了一组摄像头应用接口（Webcam API），用以快速搭建简单的姿态估计应用。在这篇教程中，我们将介绍摄像头应用接口的功能和使用方法。用户也可以在 [接口文档](https://mmpose.org.readthedocs.build/zh_cn/latest/api.html#mmpose-apis-webcam) 中查询详细信息。

<!-- TOC -->

- [摄像头应用接口概览](#摄像头应用接口概览)
- [摄像头应用示例](#摄像头应用示例)
  - [运行程序](#运行程序)
  - [配置文件](#配置文件)
    - [缓存器配置](#缓存器配置)
    - [热键配置](#热键配置)
  - [应用程序概览](#应用程序概览)
- [通过定义新节点扩展功能](#通过定义新节点扩展功能)
  - [定义一般的节点类：以目标检测为例](#定义一般的节点类以目标检测为例)
    - [继承节点基类 Node](#继承节点基类-node)
    - [定义 \_\_init\_\_() 方法](#定义-init-方法)
    - [定义 process() 方法](#定义-process-方法)
    - [定义 bypass() 方法](#定义-bypass-方法)
  - [定义可视化节点类：以文字显示为例](#定义可视化节点类以文字显示为例)
    - [继承可视化节点基类 BaseVisualizerNode](#继承可视化节点基类-basevisualizernode)
    - [实现 draw() 方法](#实现-draw-方法)

<!-- TOC -->

## 摄像头应用接口概览

<div align="center">
  <img src="https://user-images.githubusercontent.com/15977946/171577402-b28e03fa-81bd-4711-9fb8-77bcd706f7c4.png">
</div>
<div align="center">
图 1. MMPose 摄像头应用接口概览
</div>

摄像头应用接口主要由以下模块组成（如图 1 所示）：

1. **执行器**（`WebcamExecutor`）：应用程序的主体，提供了启动程序、读取视频、显示输出等基本功能。此外，在执行器中会根据配置构建若干功能模块，分别执行不同的任务，如模型推理、数据处理、逻辑判断、图像绘制等。这些模块会在不同线程中异步运行。执行器在读入视频数据后，会控制数据在各个功能模块间的流动，最终将处理完毕的视频数据显示在屏幕上。与执行器相关的概念还有：

   1. **配置文件**（`Config`）：包括执行器的基本参数，功能模块组成和模块间的逻辑关系。与 OpenMMLab 中常用的配置文件形式类似，我们使用 python 文件作为配置文件；
   2. **启动脚本**（如 `mmpose/demo/webcam_demo.py`）：读取配置文件，构建执行器类实例，并调用执行器接口启动应用程序。

2. **节点** (`Node`)：应用程序中的功能模块。一个节点通常用于实现一个基本功能，如 `DetectorNode` 用于实现目标检测，`ObjectVisualizerNode` 用于绘制图像中物体的检测框和关键点，`RecorderNode` 用于将视频输出保存到本地文件等。目前已经提供的节点根据功能可以分为模型节点（Model Node）、可视化节点（Visualizer Node）和辅助节点（Helper Node）。用户也可以根据需要，通过继承节点基类 [Node](/mmpose/apis/webcam/nodes/node.py) 实现自定义的节点。

3. **公共组件**（`Utils`）：提供了一些公共的底层模块，其中比较重要的有：

   1. **消息类**（`Message`）：节点输入、输出的基本数据结构，其中可以包括图像、模型推理结果、文本以及用户自定义的数据内容；
   2. **消息缓存器**（`Buffer`）：用于节点间的数据通信。由于各个节点是异步运行的，需要将上游节点的输出缓存下来，等待下游节点进行读取。换言之，节点之间并不直接进行数据交换，而是从指定的消息缓存器读入数据，并将输出到其他消息缓存器；
   3. **事件**（`Event`）：用于执行器与节点间或不同节点间的事件通信。与数据流通过配置好的路线依次经过节点不同，事件可以立即被任意节点响应。例如，当用户按下键盘按键时，执行器会将该事件广播给所有节点，对应节点可以立即做出响应，从而实现与用户的互动。

## 摄像头应用示例

在了解摄像头应用接口的基本组成后，我们通过一个简单的例子，来介绍如何使用这组接口搭建应用程序。

### 运行程序

通过以下指令，可以启动示例程序。它的作用是打开摄像头，将读取到的视频显示在屏幕上，同时将其保存在本地文件中：

```shell
# python demo/webcam_demo.py --config CONFIG_PATH [--debug]
python demo/webcam_demo.py --config demo/webcam_cfg/test_camera.py
```

### 配置文件

这个示例的功能模块如下所示：

```python
# 摄像头应用配置
executor_cfg = dict(
    name='Test Webcam', # 应用名称
    camera_id=0,  # 摄像头 ID（也可以用本地文件路径作为视频输入）
    camera_max_fps=30,  # 读取视频的最大帧率
    nodes=[
        # MonitorNode 用于显示系统和节点信息
        dict(
            type='MonitorNode',  # 节点类型
            name='monitor',  # 节点名称
            enable_key='m',  # 开/关快捷键
            enable=False,  # 初始开/关状态
            input_buffer='_frame_',  # 输入缓存
            output_buffer='display'),  # 输出缓存
        # RecorderNode 用于将视频保存到本地文件
        dict(
            type='RecorderNode',  # 节点类型
            name='recorder',  # 节点名称
            out_video_file='webcam_output.mp4',  # 保存视频的路径
            input_buffer='display',  # 输入缓存
            output_buffer='_display_') # 输出缓存
    ])
```

可以看到，配置文件包含一个叫做 `executor_cfg` 的字典，其中的内容包括基本参数（如 `name`，`camera_id` 等，可以参考 [WebcamExecutor 文档](https://mmpose.readthedocs.build/en/latest/generated/mmpose.apis.webcam.WebcamExecutor.html#mmpose.apis.webcam.WebcamExecutor)）和节点配置信息（`nodes` ）。节点配置信息是一个列表，其中每个元素是一个节点的参数字典。例如，在该示例中包括 2 个节点，类型分别是 `MonitorNode` 和 `RecorderNode`。关于节点的功能和参数，可以参考 [节点文档](https://mmpose.readthedocs.build/en/latest/api.html#nodes)。

#### 缓存器配置

在节点配置中有一类特殊参数——输入输出缓存器。这些配置定义了节点之间的上下游逻辑关系。比如这个例子中，`MonitorNode` 会从名为 `"_frame_"` 的缓存器中读取数据，并将输出存放到名为 `"display"` 的缓存器中；而 `RecorderNode` 则从缓存器 `"display"` 中读取数据，并将输出存放到缓存器 `"_display_"` 中。

在配置文件中，用户可以任意指定缓存器的名字，执行器会自动构建缓存器，并将缓存器与对应的节点建立关联。需要注意的是，以下 3 个是保留的缓存器名字，对应用于执行器与节点间数据交换的特殊缓存器：

- `"_input_"`：存放执行器读入的视频帧，通常用于模型推理
- `"_frame_"`：存放执行器读入的视频帧（与 `"_input_"` 相同），通常用于可视化输出
- `"_display_"`：存放经过所有节点处理后的输出结果，用于执行器在屏幕上的显示

在应用程序中，所有的缓存器是由执行器中的 **缓存管理器（BufferManager）** 进行管理（可参考 [缓存管理器文档](https://mmpose.readthedocs.build/en/latest/generated/mmpose.apis.webcam.utils.BufferManager.html#mmpose.apis.webcam.utils.BufferManager)）。

#### 热键配置

在程序运行时，部分节点（如 `MonitorNode`）可以通过键盘按键，实时控制是否生效。这类节点有以下参数：

- `enable_key`：指定开/关节点功能的热键
- `enable`：指定节点开/关的初始状态

热键响应是通过事件机制实现的。应用程序中的事件，由执行器中的 **事件管理器（EventManager）** 进行管理（可参考 [事件管理器文档](https://mmpose.readthedocs.build/en/latest/generated/mmpose.apis.webcam.utils.EventManager.html#mmpose.apis.webcam.utils.EventManager)）。节点在初始化时，可以注册与自己相关的事件，之后就可以在运行过程中触发、等待或清除这些事件。

### 应用程序概览

在了解执行器、节点、缓存器、事件等基本概念后，我们可以用图 2 概括一个基于摄像头应用接口开发的应用程序的基本结构。

<div align="center">
  <img src="https://user-images.githubusercontent.com/15977946/171552368-f961dc13-cc70-4960-bbfd-5ec791cf3b9b.png">
</div>
<div align="center">
图 2. MMPose 摄像头应用程序示意
</div>

## 通过定义新节点扩展功能

用户可以通过定义新的节点类，来扩展摄像头应用接口的功能。我们通过一些节点类实例，介绍自定义节点类的方法。

### 定义一般的节点类：以目标检测为例

我们以实现目标检测功能为例，介绍实现节点类的一般步骤和注意事项。

#### 继承节点基类 `Node`

在定义节点类时，需要继承基类 [`Node`](/mmpose/apis/webcam/nodes/node.py)，并用注册器 [`NODES`](/mmpose/apis/webcam/nodes/registry.py) 注册新的节点类，使其可以通过配置参数构建实例。

```python
from mmpose.apis.webcam.nodes import Node, NODES

@NODES.register_module()
class DetectorNode(Node):
    ...
```

#### 定义 `__init__()` 方法

我们为 `DetectorNode` 类定义以下的初始化方法，代码如下：

```python
    def __init__(self,
                 name: str,
                 model_config: str,
                 model_checkpoint: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 device: str = 'cuda:0',
                 bbox_thr: float = 0.5):

        # 初始化基类
        super().__init__(name=name, enable_key=enable_key, enable=enable)

        # 初始化节点参数
        self.model_config = get_config_path(model_config, 'mmdet')
        self.model_checkpoint = model_checkpoint
        self.device = device.lower()
        self.bbox_thr = bbox_thr

        self.model = init_detector(
            self.model_config, self.model_checkpoint, device=self.device)

        # 注册输入、输出缓存器
        self.register_input_buffer(input_buffer, 'input', trigger=True)  # 设为触发器
        self.register_output_buffer(output_buffer)
```

可以看出，初始化方法中一般会执行以下操作：

1. **初始化基类**：通常需要 `name`，`enable_key` 和 `enable` 等参数；
2. **初始化节点参数**：根据需要初始化子类的参数，如在本例中，对 `model_config`，`device`，`bbox_thr` 等参数进行了初始化，并调用 MMDetection 中的 `init_detector` API 加载了检测模型；
3. **注册缓存器**：节点收到配置参数中指定的输入、输出缓存器名称后，需要在初始化时对相应的缓存器进行注册，从而能在程序运行时自动从缓存器中存取数据。具体的方法是：
   1.对每个输入缓存器，调用基类的 `register_input_buffer()` 方法进行注册，将缓存器名（即来自配置文件的 `input_buffer` 参数）对应到一个输入名（即 `"input"`）。完成注册后，就可以在运行时通过输入名访问对应缓存器的数据（详见下一节 ["定义 process() 方法"](#定义-process-方法)）；
   2\. 调用基类的 `register_output_buffer()` 对所有的输出缓存器进行注册。完成注册后，节点在运行时每次的输出会被自动存入所有的输出缓存器（每个缓存器会存入一份输出的深拷贝）。

当节点有多个输入时，由于输入数据是异步到达的，需要指定至少一个输出缓存器为触发器（Trigger）。当所有被设为触发器的输入缓存器都有数据到达时，会触发节点执行一次操作（即执行一次 `process()` 方法）。当节点只有一个输入时，也应在注册时将其显示设置为触发器。

#### 定义 `process()` 方法

节点类的 `process()` 方法定义了节点的行为。我们在 `DetectorNode` 的 `process()` 方法中实现目标检测模型的推理过程：

```python
    def process(self, input_msgs):

        # 根据输入名 "input"，从输入缓存器获取数据
        input_msg = input_msgs['input']

        # 从输入数据中获取视频帧图像
        img = input_msg.get_image()

        # 使用 MMDetection API 进行检测模型推理
        preds = inference_detector(self.model, img)
        objects = self._post_process(preds)

        # 将目标检测结果存入数据
        input_msg.update_objects(objects)

        # 返回节点处理结果
        return input_msg
```

这段代码主要完成了以下操作：

1. **读取输入数据**：`process()` 方法的参数 `input_msgs` 中包含所有已注册输入缓存器的数据，可以通过输入名（如 `"input"`）获得对应缓存器的数据；
2. **解析输入数据**：缓存器中的数据通常为“帧信息”（`FrameMessage`，详见[文档](https://mmpose.readthedocs.build/en/latest/generated/mmpose.apis.webcam.utils.FrameMessage.html#mmpose.apis.webcam.utils.FrameMessage)）。节点可以从中获取视频帧的图像，模型推理结果等信息。
3. **处理输入数据**：这里使用 MMDetection 中的 `inference_detector()` API 检测视频帧中的物体，并进行后处理（略）。
4. **返回结果**： 将模型推理得到的视频帧中的物体信息，通过 `update_objects()` 方法添加进 `input_msg` 中，并将其返回。该结果会被自动发送到 `DetectorNode` 的所有输出缓存器，供下游节点读取。

#### 定义 `bypass()` 方法

由于允许通过热键开关 `DetectorNode` 的功能，因此需要实现 `bypass()` 方法，定义该节点在处于关闭状态时的行为。`bypass()` 方法与 `process()` 方法的接口完全相同。`DetectorNode` 在关闭时不需要对输入做任何处理，因此对 `bypass()` 实现如下：

```python
    def bypass(self, input_msgs):
        return input_msgs['input']
```

### 定义可视化节点类：以文字显示为例

可视化节点是一类特殊的节点，它们的功能是对视频帧的图像进行编辑。为了方便拓展可视化功能，我们为可视化节点提供了更简单的抽象接口。这里以实现文字显示为例，介绍实现可视化节点的一般步骤和注意事项。

#### 继承可视化节点基类 `BaseVisualizerNode`

可视化节点基类 `BaseVisualizerNode` 继承自 `Node` 类，并对 `process()` 方法进行了进一步封装，暴露 `draw()` 接口供子类实现具体的可视化功能。与一般的节点类类似，可视化节点类需要继承 `BaseVisualizerNode` 并注册进 `NODES`。

```python
from mmpose.apis.webcam.nodes import BaseVisualizerNode, NODES

@NODES.register_module()
class NoticeBoardNode(BaseVisualizerNode):
    ...
```

节点初始化方法的实现方式与一般节点类似，请参考 [定义 \_\_init\_\_() 方法](#定义-init-方法)。需要注意的是，可视化节点必须注册唯一的输入缓存器，对应于输入名 `"input"`。

#### 实现 `draw()` 方法

可视化节点类的 `draw()` 方法用于绘制对帧图像的更新。`draw()` 方法有 1 个输入参数 `input_msg`，为来自 `input` 缓存器的数据；`draw()` 方法的返回值应为一幅图像（`np.ndarray` 类型），该图像将被用于更新 `input_msg` 中的图像。节点的输出即为更新图像后的 `input_msg`。

`NoticeBoardNode` 的 `draw()` 方法实现如下：

```python
    def draw(self, input_msg: FrameMessage) -> np.ndarray:
        # 获取帧图像
        img = input_msg.get_image()

        # 创建画布图像
        canvas = np.full(img.shape, self.background_color, dtype=img.dtype)

        # 逐行将文字绘制在画布图像上
        x = self.x_offset
        y = self.y_offset
        max_len = max([len(line) for line in self.content_lines])

        def _put_line(line=''):
            nonlocal y
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        self.text_scale, self.text_color, 1)
            y += self.y_delta

        for line in self.content_lines:
            _put_line(line)

        # 将画布图像的有效区域叠加在帧图像上
        x1 = max(0, self.x_offset)
        x2 = min(img.shape[1], int(x + max_len * self.text_scale * 20))
        y1 = max(0, self.y_offset - self.y_delta)
        y2 = min(img.shape[0], y)

        src1 = canvas[y1:y2, x1:x2]
        src2 = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

        # 返回绘制结果
        return img
```
