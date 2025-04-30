# 使用现有模型进行推理

MMPose 为姿态估计提供了大量可以从 [模型库](https://mmpose.readthedocs.io/en/latest/model_zoo.html) 中找到的预测训练模型。

本指南将演示**如何执行推理**，或使用训练过的模型对提供的图像或视频运行姿态估计。

MMPose 提供了两种推理接口：

1. 推理器：统一的推理接口
2. 推理 API：用于更加灵活的自定义推理

## 推理器：统一的推理接口

MMPose 提供了一个被称为 [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) 的、全面的推理 API。这个 API 使得用户得以使用所有 MMPose 支持的模型来对图像和视频进行模型推理。此外，该API可以完成推理结果自动化，并方便用户保存预测结果。

### 基本用法

[MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) 可以在任何 Python 程序中被用来执行姿态估计任务。以下是在一个在 Python Shell 中使用预训练的人体姿态模型对给定图像进行推理的示例。

```python
from mmpose.apis import MMPoseInferencer

img_path = 'tests/data/coco/000000000785.jpg'   # 将img_path替换给你自己的路径

# 使用模型别名创建推理器
inferencer = MMPoseInferencer('human')

# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
```

如果一切正常，你将在一个新窗口中看到下图：

![inferencer_result_coco](https://user-images.githubusercontent.com/26127467/220008302-4a57fd44-0978-408e-8351-600e5513316a.jpg)

`result` 变量是一个包含两个键值 `'visualization'` 和 `'predictions'` 的字典。

- `'visualization'` 键对应的值是一个列表，该列表：
  - 包含可视化结果，例如输入图像、估计姿态的标记，以及可选的预测热图。
  - 如果没有指定 `return_vis` 参数，该列表将保持为空。
- `'predictions'` 键对应的值是：
  - 一个包含每个检测实例的预估关键点的列表。

`result` 字典的结构如下所示：

```python
result = {
    'visualization': [
        # 元素数量：batch_size（默认为1）
        vis_image_1,
        ...
    ],
    'predictions': [
        # 每张图像的姿态估计结果
        # 元素数量：batch_size（默认为1）
        [
            # 每个检测到的实例的姿态信息
            # 元素数量：检测到的实例数
            {'keypoints': ...,  # 实例 1
            'keypoint_scores': ...,
            ...
            },
            {'keypoints': ...,  # 实例 2
            'keypoint_scores': ...,
            ...
            },
        ]
    ...
    ]
}
```

还可以使用用于用于推断的**命令行界面工具**（CLI, command-line interface）: `demo/inferencer_demo.py`。这个工具允许用户使用以下命令使用相同的模型和输入执行推理：

```python
python demo/inferencer_demo.py 'tests/data/coco/000000000785.jpg' \
    --pose2d 'human' --show --pred-out-dir 'predictions'
```

预测结果将被保存在路径 `predictions/000000000785.json` 。作为一个API，`inferencer_demo.py` 的输入参数与 [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) 的相同。前者能够处理一系列输入类型，包括以下内容：

- 图像路径

- 视频路径

- 文件夹路径（这会导致该文件夹中的所有图像都被推断出来）

- 表示图像的 numpy array (在命令行界面工具中未支持)

- 表示图像的 numpy array 列表 (在命令行界面工具中未支持)

- 摄像头（在这种情况下，输入参数应该设置为 `webcam` 或 `webcam:{CAMERA_ID}`）

当输入对应于多个图像时，例如输入为**视频**或**文件夹**路径时，推理生成器必须被遍历，以便推理器对视频/文件夹中的所有帧/图像进行推理。以下是一个示例：

```python
folder_path = 'tests/data/coco'

result_generator = inferencer(folder_path, show=True)
results = [result for result in result_generator]
```

在这个示例中，`inferencer` 接受 `folder_path` 作为输入，并返回一个生成器对象（`result_generator`），用于生成推理结果。通过遍历 `result_generator` 并将每个结果存储在 `results` 列表中，您可以获得视频/文件夹中所有帧/图像的推理结果。

### 自定义姿态估计模型

[MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) 提供了几种可用于自定义所使用的模型的方法：

```python
# 使用模型别名构建推理器
inferencer = MMPoseInferencer('human')

# 使用模型配置名构建推理器
inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')
# 使用 3D 模型配置名构建推理器
inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")

# 使用模型配置文件和权重文件的路径或 URL 构建推理器
inferencer = MMPoseInferencer(
    pose2d='configs/body_2d_keypoint/topdown_heatmap/coco/' \
           'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    pose2d_weights='https://download.openmmlab.com/mmpose/top_down/' \
                   'hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
)
```

模型别名的完整列表可以在模型别名部分中找到。

上述代码为 2D 模型推理器的构建例子。3D 模型的推理器可以用类似的方式通过 `pose3d` 参数构建：

```python
# 使用 3D 模型别名构建推理器
inferencer = MMPoseInferencer(pose3d="human3d")

# 使用 3D 模型配置名构建推理器
inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")

# 使用 3D 模型配置文件和权重文件的路径或 URL 构建推理器
inferencer = MMPoseInferencer(
    pose3d='configs/body_3d_keypoint/motionbert/h36m/' \
           'motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py',
    pose3d_weights='https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/' \
                   'pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth'
)
```

此外，自顶向下的姿态估计器还需要一个对象检测模型。[MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) 能够推断用 MMPose 支持的数据集训练的模型的实例类型，然后构建必要的对象检测模型。用户也可以通过以下方式手动指定检测模型:

```python
# 通过别名指定检测模型
# 可用的别名包括“human”、“hand”、“face”、“animal”、
# 以及mmdet中定义的任何其他别名
inferencer = MMPoseInferencer(
    # 假设姿态估计器是在自定义数据集上训练的
    pose2d='custom_human_pose_estimator.py',
    pose2d_weights='custom_human_pose_estimator.pth',
    det_model='human'
)

# 使用模型配置名称指定检测模型
inferencer = MMPoseInferencer(
    pose2d='human',
    det_model='yolox_l_8x8_300e_coco',
    det_cat_ids=[0],  # 指定'human'类的类别id
)

# 使用模型配置文件和权重文件的路径或URL构建推理器
inferencer = MMPoseInferencer(
    pose2d='human',
    det_model=f'{PATH_TO_MMDET}/configs/yolox/yolox_l_8x8_300e_coco.py',
    det_weights='https://download.openmmlab.com/mmdetection/v2.0/' \
                'yolox/yolox_l_8x8_300e_coco/' \
                'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    det_cat_ids=[0],  # 指定'human'类的类别id
)
```

### 转储结果

在执行姿态估计推理任务之后，您可能希望保存结果以供进一步分析或处理。本节将指导您将预测的关键点和可视化结果保存到本地。

要将预测保存在<mark>JSON文件</mark>中，在运行 [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) 的实例 `inferencer` 时使用 `pred_out_dir` 参数:

```python
result_generator = inferencer(img_path, pred_out_dir='predictions')
result = next(result_generator)
```

预测结果将以 JSON 格式保存在 `predictions/` 文件夹中，每个文件以相应的输入图像或视频的名称命名。

对于更高级的场景，还可以直接从 `inferencer` 返回的 `result` 字典中访问预测结果。其中，`predictions` 包含输入图像或视频中每个单独实例的预测关键点列表。然后，您可以使用您喜欢的方法操作或存储这些结果。

请记住，如果你想将<mark>可视化图像</mark>和预测文件保存在一个文件夹中，你可以使用 `out_dir` 参数：

```python
result_generator = inferencer(img_path, out_dir='output')
result = next(result_generator)
```

在这种情况下，可视化图像将保存在 `output/visualization/` 文件夹中，而预测将存储在 `output/forecasts/` 文件夹中。

### 可视化

推理器 `inferencer` 可以自动对输入的图像或视频进行预测。可视化结果可以显示在一个新的窗口中，并保存在本地。

要在新窗口中查看可视化结果，请使用以下代码：

请注意：

- 如果输入视频来自网络摄像头，默认情况下将在新窗口中显示可视化结果，以此让用户看到输入

- 如果平台上没有 GUI，这个步骤可能会卡住

要将可视化结果保存在本地，可以像这样指定`vis_out_dir`参数:

```python
result_generator = inferencer(img_path, vis_out_dir='vis_results')
result = next(result_generator)
```

输入图片或视频的可视化预测结果将保存在 `vis_results/` 文件夹中

在开头展示的滑雪图中，姿态的可视化估计结果由关键点（用实心圆描绘）和骨架（用线条表示）组成。这些视觉元素的默认大小可能不会产生令人满意的结果。用户可以使用 `radius` 和 `thickness` 参数来调整圆的大小和线的粗细，如下所示：

```python
result_generator = inferencer(img_path, show=True, radius=4, thickness=2)
result = next(result_generator)
```

### 推理器参数

[MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) 提供了各种自定义姿态估计、可视化和保存预测结果的参数。下面是<mark>初始化</mark>推理器时可用的参数列表及对这些参数的描述：

| Argument         | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `pose2d`         | 指定 2D 姿态估计模型的模型别名、配置文件名称或配置文件路径。 |
| `pose2d_weights` | 指定 2D 姿态估计模型权重文件的URL或本地路径。                |
| `pose3d`         | 指定 3D 姿态估计模型的模型别名、配置文件名称或配置文件路径。 |
| `pose3d_weights` | 指定 3D 姿态估计模型权重文件的URL或本地路径。                |
| `det_model`      | 指定对象检测模型的模型别名、配置文件名或配置文件路径。       |
| `det_weights`    | 指定对象检测模型权重文件的 URL 或本地路径。                  |
| `det_cat_ids`    | 指定与要检测的对象类对应的类别 id 列表。                     |
| `device`         | 执行推理的设备。如果为 `None`，推理器将选择最合适的一个。    |
| `scope`          | 定义模型模块的名称空间                                       |

推理器被设计用于可视化和保存预测。以下表格列出了在使用 [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) <mark>进行推断</mark>时可用的参数列表，以及它们与 2D 和 3D 推理器的兼容性：

| 参数                      | 描述                                                                                                                       | 2D  | 3D  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --- | --- |
| `show`                    | 控制是否在弹出窗口中显示图像或视频。                                                                                       | ✔️  | ✔️  |
| `radius`                  | 设置可视化关键点的半径。                                                                                                   | ✔️  | ✔️  |
| `thickness`               | 确定可视化链接的厚度。                                                                                                     | ✔️  | ✔️  |
| `kpt_thr`                 | 设置关键点分数阈值。分数超过此阈值的关键点将被显示。                                                                       | ✔️  | ✔️  |
| `draw_bbox`               | 决定是否显示实例的边界框。                                                                                                 | ✔️  | ✔️  |
| `draw_heatmap`            | 决定是否绘制预测的热图。                                                                                                   | ✔️  | ❌  |
| `black_background`        | 决定是否在黑色背景上显示预估的姿势。                                                                                       | ✔️  | ❌  |
| `skeleton_style`          | 设置骨架样式。可选项包括 'mmpose'（默认）和 'openpose'。                                                                   | ✔️  | ❌  |
| `use_oks_tracking`        | 决定是否在追踪中使用OKS作为相似度测量。                                                                                    | ❌  | ✔️  |
| `tracking_thr`            | 设置追踪的相似度阈值。                                                                                                     | ❌  | ✔️  |
| `disable_norm_pose_2d`    | 决定是否将边界框缩放至数据集的平均边界框尺寸，并将边界框移至数据集的平均边界框中心。                                       | ❌  | ✔️  |
| `disable_rebase_keypoint` | 决定是否将最低关键点的高度置为 0。                                                                                         | ❌  | ✔️  |
| `num_instances`           | 设置可视化结果中显示的实例数量。如果设置为负数，则所有实例的结果都会可视化。                                               | ❌  | ✔️  |
| `return_vis`              | 决定是否在结果中包含可视化图像。                                                                                           | ✔️  | ✔️  |
| `vis_out_dir`             | 定义保存可视化图像的文件夹路径。如果未设置，将不保存可视化图像。                                                           | ✔️  | ✔️  |
| `return_datasamples`      | 决定是否以 `PoseDataSample` 格式返回预测。                                                                                 | ✔️  | ✔️  |
| `pred_out_dir`            | 指定保存预测的文件夹路径。如果未设置，将不保存预测。                                                                       | ✔️  | ✔️  |
| `out_dir`                 | 如果 `vis_out_dir` 或 `pred_out_dir` 未设置，它们将分别设置为 `f'{out_dir}/visualization'` 或 `f'{out_dir}/predictions'`。 | ✔️  | ✔️  |

### 模型别名

MMPose 为常用模型提供了一组预定义的别名。在初始化 [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) 时，这些别名可以用作简略的表达方式，而不是指定完整的模型配置名称。下面是可用的模型别名及其对应的配置名称的列表：

| 别名      | 配置文件名称                                       | 对应任务         | 姿态估计模型  | 检测模型            |
| --------- | -------------------------------------------------- | ---------------- | ------------- | ------------------- |
| animal    | rtmpose-m_8xb64-210e_ap10k-256x256                 | 动物姿态估计     | RTMPose-m     | RTMDet-m            |
| human     | rtmpose-m_8xb256-420e_body8-256x192                | 人体姿态估计     | RTMPose-m     | RTMDet-m            |
| body26    | rtmpose-m_8xb512-700e_body8-halpe26-256x192        | 人体姿态估计     | RTMPose-m     | RTMDet-m            |
| face      | rtmpose-m_8xb256-120e_face6-256x256                | 人脸关键点检测   | RTMPose-m     | yolox-s             |
| hand      | rtmpose-m_8xb256-210e_hand5-256x256                | 手部关键点检测   | RTMPose-m     | ssdlite_mobilenetv2 |
| wholebody | rtmpose-m_8xb64-270e_coco-wholebody-256x192        | 人体全身姿态估计 | RTMPose-m     | RTMDet-m            |
| vitpose   | td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192  | 人体姿态估计     | ViTPose-base  | RTMDet-m            |
| vitpose-s | td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192 | 人体姿态估计     | ViTPose-small | RTMDet-m            |
| vitpose-b | td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192  | 人体姿态估计     | ViTPose-base  | RTMDet-m            |
| vitpose-l | td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192 | 人体姿态估计     | ViTPose-large | RTMDet-m            |
| vitpose-h | td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192  | 人体姿态估计     | ViTPose-huge  | RTMDet-m            |

下表列出了可用的 3D 姿态估计模型别名及其对应的配置文件：

| 别名    | 配置文件名称                                 | 对应任务          | 3D 姿态估计模型 | 2D 姿态估计模型 | 检测模型 |
| ------- | -------------------------------------------- | ----------------- | --------------- | --------------- | -------- |
| human3d | vid_pl_motionbert_8xb32-120e_h36m            | 3D 人体姿态估计   | MotionBert      | RTMPose-m       | RTMDet-m |
| hand3d  | internet_res50_4xb16-20e_interhand3d-256x256 | 3D 手部关键点检测 | InterNet        | -               | 全图     |

此外，用户可以使用命令行界面工具显示所有可用的别名，使用以下命令:

```shell
python demo/inferencer_demo.py --show-alias
```

## 推理 API：用于更加灵活的自定义推理

MMPose 提供了单独的 Python API 用于不同模型的推理，这种推理方式更加灵活，但是需要用户自己处理输入和输出，因此适合于**熟悉 MMPose** 的用户。

MMPose 提供的 Python 推理接口存放于 [$MMPOSE/mmpose/apis](https://github.com/open-mmlab/mmpose/tree/dev-1.x/mmpose/apis) 目录下，以下是一个构建 topdown 模型并进行推理的示例：

### 构建模型

```python
from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

model_cfg = 'configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'

ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth'

device = 'cuda'

# 使用初始化接口构建模型
model = init_model(model_cfg, ckpt, device=device)
```

### 推理

```python
img_path = 'tests/data/coco/000000000785.jpg'

# 单张图片推理
batch_results = inference_topdown(model, img_path)
```

推理接口返回的结果是一个 PoseDataSample 列表，每个 PoseDataSample 对应一张图片的推理结果。PoseDataSample 的结构如下所示：

```python
[
    <PoseDataSample(

        ori_shape: (425, 640)
        img_path: 'tests/data/coco/000000000785.jpg'
        input_size: (192, 256)
        flip_indices: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        img_shape: (425, 640)

        gt_instances: <InstanceData(
                bboxes: array([[  0.,   0., 640., 425.]], dtype=float32)
                bbox_centers: array([[320. , 212.5]], dtype=float32)
                bbox_scales: array([[ 800.    , 1066.6666]], dtype=float32)
                bbox_scores: array([1.], dtype=float32)
            )>

        gt_instance_labels: <InstanceData()>

        pred_instances: <InstanceData(
                keypoints: array([[[365.83333333,  87.50000477],
                            [372.08333333,  79.16667175],
                            [361.66666667,  81.25000501],
                            [384.58333333,  85.41667151],
                            [357.5       ,  85.41667151],
                            [407.5       , 112.50000381],
                            [363.75      , 125.00000334],
                            [438.75      , 150.00000238],
                            [347.08333333, 158.3333354 ],
                            [451.25      , 170.83333492],
                            [305.41666667, 177.08333468],
                            [432.5       , 214.58333325],
                            [401.25      , 218.74999976],
                            [430.41666667, 285.41666389],
                            [370.        , 274.99999762],
                            [470.        , 356.24999452],
                            [403.33333333, 343.74999499]]])
                bbox_scores: array([1.], dtype=float32)
                bboxes: array([[  0.,   0., 640., 425.]], dtype=float32)
                keypoint_scores: array([[0.8720184 , 0.9068178 , 0.89255375, 0.94684595, 0.83111566,
                            0.9929208 , 1.0862956 , 0.9265839 , 0.9781244 , 0.9008082 ,
                            0.9043166 , 1.0150217 , 1.1122335 , 1.0207931 , 1.0099326 ,
                            1.0480015 , 1.0897669 ]], dtype=float32)
                keypoints_visible: array([[0.8720184 , 0.9068178 , 0.89255375, 0.94684595, 0.83111566,
                            0.9929208 , 1.0862956 , 0.9265839 , 0.9781244 , 0.9008082 ,
                            0.9043166 , 1.0150217 , 1.1122335 , 1.0207931 , 1.0099326 ,
                            1.0480015 , 1.0897669 ]], dtype=float32)
            )>
    )>
]
```

用户可以通过 `.` 来访问 PoseDataSample 中的数据，例如：

```python
pred_instances = batch_results[0].pred_instances

pred_instances.keypoints
# array([[[365.83333333,  87.50000477],
#         [372.08333333,  79.16667175],
#         [361.66666667,  81.25000501],
#         [384.58333333,  85.41667151],
#         [357.5       ,  85.41667151],
#         [407.5       , 112.50000381],
#         [363.75      , 125.00000334],
#         [438.75      , 150.00000238],
#         [347.08333333, 158.3333354 ],
#         [451.25      , 170.83333492],
#         [305.41666667, 177.08333468],
#         [432.5       , 214.58333325],
#         [401.25      , 218.74999976],
#         [430.41666667, 285.41666389],
#         [370.        , 274.99999762],
#         [470.        , 356.24999452],
#         [403.33333333, 343.74999499]]])
```

### 可视化

在 MMPose 中，大部分可视化基于可视化器实现。可视化器是一个类，它接受数据样本并将其可视化。MMPose 提供了一个可视化器注册表，用户可以使用 `VISUALIZERS` 来实例化它。以下是一个使用可视化器可视化推理结果的示例：

```python
# 将推理结果打包
results = merge_data_samples(batch_results)

# 初始化可视化器
visualizer = VISUALIZERS.build(model.cfg.visualizer)

# 设置数据集元信息
visualizer.set_dataset_meta(model.dataset_meta)

img = imread(img_path, channel_order='rgb')

# 可视化
visualizer.add_datasample(
    'result',
    img,
    data_sample=results,
    show=True)
```

MMPose 也提供了更简洁的可视化接口：

```python
from mmpose.apis import visualize

pred_instances = batch_results[0].pred_instances

keypoints = pred_instances.keypoints
keypoint_scores = pred_instances.keypoint_scores

metainfo = 'config/_base_/datasets/coco.py'

visualize(
    img_path,
    keypoints,
    keypoint_scores,
    metainfo=metainfo,
    show=True)
```
