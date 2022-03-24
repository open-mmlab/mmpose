# 开发示例：给猫咪戴上太阳镜

## 设计思路

在动手之前，我们先考虑如何实现这个功能：

- 首先，要做目标检测，找到图像中的猫咪
- 接着，要估计猫咪的关键点位置，比如左右眼的位置
- 最后，把太阳镜素材图片贴在合适的位置，TA-DA！

按照这个思路，下面我们来看如何一步一步实现它。

## Step 1：从一个现成的 Config 开始

在 WebcamAPI 中，已经添加了一些实现常用功能的 Node，并提供了对应的 config 示例。利用这些可以减少用户的开发量。例如，我们可以以上面的姿态估计 demo 为基础。它的 config 位于 `tools/webcam/configs/example/pose_estimation.py`。为了更直观，我们把这个 config 中的功能节点表示成以下流程图：

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/153801397-640f2b45-64e7-41b3-8b00-670c16c57df5.png">
</div>
<div align=center>
Pose Estimation Config 示意
</div>

可以看到，这个 config 已经实现了我们设计思路中“1-目标检测”和“2-关键点检测”的功能。我们还需要实现“3-贴素材图”功能，这就需要定义一个新的 Node了。

## Step 2：实现一个新 Node

在 WebcamAPI 我们定义了以下 2 个 Node 基类：

1. Node：所有 node 的基类，实现了初始化，绑定 runner，启动运行，数据输入输出等基本功能。子类通过重写抽象方法`process()`方法定义具体的 node 功能。
2. FrameDrawingNode：用来绘制图像的 node 基类。FrameDrawingNode继承自 Node 并进一步封装了`process()`方法，提供了抽象方法`draw()`供子类实现具体的图像绘制功能。

显然，“贴素材图”这个功能属于图像绘制，因此我们只需要继承 FrameDrawingNode 类即可。具体实现如下：

```python
# 假设该文件路径为
# <MMPose Root>/tools/webcam/webcam_apis/nodes/sunglasses_node.py
from mmpose.core import apply_sunglasses_effect
from ..utils import (load_image_from_disk_or_url,
    get_eye_keypoint_ids)
from .frame_drawing_node import FrameDrawingNode
from .builder import NODES

@NODES.register_module()  # 将 SunglassesNode 注册到 NODES（Registry）
class SunglassesNode(FrameDrawingNode):

    def __init__(self,
                 name: str,
                 frame_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 src_img_path: Optional[str] = None):

        super().__init__(name, frame_buffer, output_buffer, enable_key, enable)

        # 加载素材图片
        if src_img_path is None:
            # The image attributes to:
            # https://www.vecteezy.com/free-vector/glass
            # Glass Vectors by Vecteezy
            src_img_path = ('https://raw.githubusercontent.com/open-mmlab/'
                            'mmpose/master/demo/resources/sunglasses.jpg')
        self.src_img = load_image_from_disk_or_url(src_img_path)

    def draw(self, frame_msg):
        # 获取当前帧图像
        canvas = frame_msg.get_image()
        # 获取姿态估计结果
        pose_results = frame_msg.get_pose_results()
        if not pose_results:
            return canvas

        # 给每个目标添加太阳镜效果
        for pose_result in pose_results:
            model_cfg = pose_result['model_cfg']
            preds = pose_result['preds']
            # 获取目标左、右眼关键点位置
            left_eye_idx, right_eye_idx = get_eye_keypoint_ids(model_cfg)
            # 根据双眼位置，绘制太阳镜
            canvas = apply_sunglasses_effect(canvas, preds, self.src_img,
                                             left_eye_idx, right_eye_idx)
        return canvas
```

这里对代码实现中用到的一些函数和类稍作说明：

1. `NODES`：是一个 mmcv.Registry 实例。相信用过 OpenMMLab 系列的同学都对 Registry 不陌生。这里用 NODES来注册和管理所有的 node 类，从而让用户可以在 config 中通过类的名称（如 "DetectorNode"，"SunglassesNode" 等）来指定使用对应的 node。
2. `load_image_from_disk_or_url`：用来从本地路径或 url 读取图片
3. `get_eye_keypoint_ids`：根据模型配置文件（model_cfg）中记录的数据集信息，返回双眼关键点的索引。如 COCO 格式对应的左右眼索引为 $(1,2)$
4. `apply_sunglasses_effect`：将太阳镜绘制到原图中的合适位置，具体步骤为：
    - 在素材图片上定义一组源锚点 $(s_1, s_2, s_3, s_4)$
    - 根据目标左右眼关键点位置 $(k_1, k_2)$，计算目标锚点 $(t_1, t_2, t_3, t_4)$
    - 通过源锚点和目标锚点，计算几何变换矩阵（平移，缩放，旋转），将素材图片做变换后贴入原图片。即可将太阳镜绘制在合适的位置。

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/153773612-bcf86b91-31a3-47b5-886d-e33577016f85.png">
</div>
太阳镜特效原理示意
<div align=center>
</div>

### Get Advanced：关于 Node 和 FrameEffectNode

[Node 类](/tools/webcam/webcam_apis/nodes/node.py) ：继承自 Thread 类。正如我们在前面 数据流 部分提到的，所有节点都在各自的线程中彼此异步运行。在`Node.run()` 方法中定义了节点的基本运行逻辑：

1. 当 buffer 中有数据时，会触发一次运行
2. 调用`process()`来执行具体的功能。`process()`是一个抽象接口，由子类具体实现
    - 特别地，如果节点需要实现“开/关”功能，则还需要实现`bypass()`方法，以定义节点“关”时的行为。`bypass()`与`process()`的输入输出接口完全相同。在run()中会根据`Node.enable`的状态，调用`process()`或`bypass()`
3. 将运行结果发送到输出 buffer

在继承 Node 类实现具体的节点类时，通常需要完成以下工作：

1. 在__init__()中注册输入、输出 buffer，并调用基类的__init__()方法
2. 实现process()和bypass()（如需要）方法

[FrameDrawingNode 类](/tools/webcam/webcam_apis/nodes/frame_drawing_node.py) ：继承自 Node 类，对`process()`和`bypass()`方法做了进一步封装：

- process()：从接到输入中提取帧图像，传入draw()方法中绘图。draw()是一个抽象接口，有子类实现
- bypass()：直接将节点输入返回

### Get Advanced: 关于节点的输入、输出格式

我们定义了[FrameMessage 类](/tools/webcam/webcam_apis/utils/message.py)作为节点间通信的数据结构。也就是说，通常情况下节点的输入、输出和 buffer 中存储的元素，都是 FrameMessage 类的实例。FrameMessage 通常用来存储视频中1帧的信息，它提供了简单的接口，用来提取和存入数据：

- `get_image()`：返回图像
- `set_image()`：设置图像
- `add_detection_result()`：添加一个目标检测模型的结果
- `get_detection_results()`：返回所有目标检测结果
- `add_pose_result()`：添加一个姿态估计模型的结果
- `get_pose_results()`：返回所有姿态估计结果

## Step 3：调整 Config

有了 Step 2 中实现的 SunglassesNode，我们只要把它加入 config 里就可以使用了。比如，我们可以把它放在“Visualizer” node 之后：

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/153801499-590a7810-b231-4a38-8053-c7d33af1535a.png">
</div>
修改后的 Config，添加了 SunglassesNode 节点
<div align=center>
</div>

具体的写法如下：

```python
runner = dict(
    # runner的基本参数
    name='Everybody Wears Sunglasses',
    camera_id=0,
    camera_fps=20,
    # 定义了若干节点(node)
    nodes=[
        ...,
        dict(
            type='SunglassesNode',  # 节点类名称
            name='Sunglasses',  # 节点名，由用户自己定义
            frame_buffer='vis',  # 输入
            output_buffer='sunglasses',  # 输出
            enable_key='s',  # 定义开关快捷键
            enable=True,)  # 启动时默认的开关状态
        ...]  # 更多节点
)
```

此外，用户还可以根据需求调整 config 中的参数。一些常用的设置包括：

1. 选择摄像头：可以通过设置camera_id参数指定使用的摄像头。通常电脑上的默认摄像头 id 为 0，如果有多个则 id 数字依次增大。此外，也可以给camera_id设置一个本地视频文件的路径，从而使用该视频文件作为应用程序的输入
2. 选择模型：可以通过模型推理节点（如 DetectorNode，TopDownPoseEstimationNode）的model_config和model_checkpoint参数来配置。用户可以根据自己的需求（如目标物体类别，关键点类别等）和硬件情况选用合适的模型
3. 设置快捷键：一些 node 支持使用快捷键开关，用户可以设置对应的enable_key（快捷键）和enable（默认开关状态）参数
4. 提示信息：通过设置 NoticeBoardNode 的 content_lines参数，可以在程序运行时在画面上显示提示信息，帮助使用者快速了解这个应用程序的功能和操作方法

最后，将修改过的 config 存到文件`tools/webcam/configs/sunglasses.py`中，就可以运行了：

```shell
python tools/webcam/run_webcam.py --config tools/webcam/configs/sunglasses.py
```
