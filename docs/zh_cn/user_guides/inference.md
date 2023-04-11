# 使用现有模型进行推理

MMPose为姿态估计提供了大量可以从[模型库](https://mmpose.readthedocs.io/en/latest/model_zoo.html)中找到的预测训练模型。本指南将演示**如何执行推理**，或使用训练过的模型对提供的图像或视频运行姿态估计。

有关在标准数据集上测试现有模型的说明，请参阅本指南。

在MMPose，模型由配置文件定义，而其已计算好的参数存储在权重文件（checkpoint file）中。您可以在[模型库](https://mmpose.readthedocs.io/en/latest/model_zoo.html)中找到模型配置文件和相应的权重文件的URL。我们建议从使用HRNet模型的[配置文件](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py)和[权重文件](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth)开始。

# 推理器：统一的推理接口

MMPose提供了一个被称为`MMPoseInferencer`的、全面的推理API。这个API使得用户得以使用所有MMPose支持的模型来对图像和视频进行模型推理。此外，该API可以完成推理结果自动化，并方便用户保存预测结果。

## 基本用法

`MMPoseInferencer`可以在任何Python程序中被用来执行姿态估计任务。以下是在一个在Python Shell中使用预训练的人体姿态模型对给定图像进行推理的示例。

```python
from mmpose.apis import MMPoseInferencer

img_path = 'tests/data/coco/000000000785.jpg'   # 将img_path替换给你自己的路径

# 使用模型别名创建推断器
inferencer = MMPoseInferencer('human')

# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
```

如果一切正常，你将在一个新窗口中看到下图：

![inferencer_result_coco](https://user-images.githubusercontent.com/26127467/220008302-4a57fd44-0978-408e-8351-600e5513316a.jpg)

在上述示例中，变量`result`是一个字典，包含两个键，分别是`visualization`和`predictions`。`visualization`用于存储可视化结果，但由于没有设定参数`return_vis`，因此该列表为空。但是`predictions`保存了每个检测到的实例的、估计得到的关键点列表。

还可以使用用于用于推断的**命令行界面工具**（CLI, command-line interface）:`demo/inferencer_demo.py`。这个工具允许用户使用以下命令使用相同的模型和输入执行推理：

```python
python demo/inferencer_demo.py 'tests/data/coco/000000000785.jpg' \
    --pose2d 'human' --show --pred-out-dir 'predictions'
```

预测结果将被保存在路径`predictions/000000000785.json`。作为一个API，`inferencer_demo.py`的输入参数与`MMPoseInferencer`的相同。前者能够处理一系列输入类型，包括以下内容：

- 图像路径

- 视频路径

- 文件夹路径（这会导致该文件夹中的所有图像都被推断出来）

- An image array (NA for CLI tool)

- A list of image arrays (NA for CLI tool)

- 摄像头（在这种情况下，输入参数应该设置为`webcam`或`webcam:{CAMERA_ID}`）

## 自定义姿态估计模型

`MMPoseInferencer`提供了几种可用于自定义所使用的模型的方法：

```python
# 使用模型别名构建推断器
inferencer = MMPoseInferencer('human')

# 使用模型配置名构建推断器
inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')

# 使用模型配置文件和权重文件的路径或URL构建推断器
inferencer = MMPoseInferencer(
    pose2d='configs/body_2d_keypoint/topdown_heatmap/coco/' \
           'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    pose2d_weights='https://download.openmmlab.com/mmpose/top_down/' \
                   'hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
)
```

模型别名的完整列表可以在模型别名部分（Model Alias section）中找到。

此外，自顶向下的姿态估计器还需要一个对象检测模型。`MMPoseInferencer`能够推断用MMPose支持的数据集训练的模型的实例类型，然后构建必要的对象检测模型。用户也可以通过以下方式手动指定检测模型:

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

# 使用模型配置文件和权重文件的路径或URL构建推断器
inferencer = MMPoseInferencer(
    pose2d='human',
    det_model=f'{PATH_TO_MMDET}/configs/yolox/yolox_l_8x8_300e_coco.py',
    det_weights='https://download.openmmlab.com/mmdetection/v2.0/' \
                'yolox/yolox_l_8x8_300e_coco/' \
                'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    det_cat_ids=[0],  # 指定'human'类的类别id
)
```

## 转储结果

在执行姿态估计推理任务之后，您可能希望保存结果以供进一步分析或处理。本节将指导您将预测的关键点和可视化结果保存到本地。

要将预测保存在<mark>JSON文件</mark>中，在运行`MMPoseInferencer`的实例`inferencer`时使用`pred_out_dir`参数:

```python
result_generator = inferencer(img_path, pred_out_dir='predictions')
result = next(result_generator)
```

预测结果将以JSON格式保存在`predictions/`文件夹中，每个文件以相应的输入图像或视频的名称命名。

对于更高级的场景，还可以直接从`inferencer`返回的`result`字典中访问预测结果。其中，`predictions`包含输入图像或视频中每个单独实例的预测关键点列表。然后，您可以使用您喜欢的方法操作或存储这些结果。

请记住，如果你想将<mark>可视化图像</mark>和预测文件保存在一个文件夹中，你可以使用`out_dir`参数：

```python
result_generator = inferencer(img_path, out_dir='output')
result = next(result_generator)
```

在这种情况下，可视化图像将保存在`output/visualization/`文件夹中，而预测将存储在`output/forecasts/`文件夹中。

## 可视化

推理器`inferencer`可以自动对输入的图像或视频进行预测。可视化结果可以显示在一个新的窗口中，并保存在本地。

要在新窗口中查看可视化结果，请使用以下代码：

请注意：

- 如果输入视频来自网络摄像头，默认情况下将在新窗口中显示可视化结果，以此让用户看到输入

- 如果平台上没有GUI，这个步骤可能会卡住

要将可视化结果保存在本地，可以像这样指定`vis_out_dir`参数:

```python
result_generator = inferencer(img_path, vis_out_dir='vis_results')
result = next(result_generator)
```

输入图片或视频的可视化预测结果将保存在`vis_results/`文件夹中

在开头展示的滑雪图中，姿态的可视化估计结果由关键点（用实心圆描绘）和骨架（用线条表示）组成。这些视觉元素的默认大小可能不会产生令人满意的结果。用户可以使用`radius`和`thickness`参数来调整圆的大小和线的粗细，如下所示：

```python
result_generator = inferencer(img_path, show=True, radius=4, thickness=2)
result = next(result_generator)
```

## 推理器参数

`MMPoseInferencer`提供了各种自定义姿态估计、可视化和保存预测结果的参数。下面是<mark>初始化</mark>推断器时可用的参数列表及对这些参数的描述：

| Argument         | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `pose2d`         | 指定2D姿态估计模型的模型别名、配置文件名称或配置文件路径。 |
| `pose2d_weights` | 指定2D姿态估计模型权重文件的URL或本地路径。                |
| `det_model`      | 指定对象检测模型的模型别名、配置文件名或配置文件路径。     |
| `det_weights`    | 指定对象检测模型权重文件的URL或本地路径。                  |
| `det_cat_ids`    | 指定与要检测的对象类对应的类别id列表。                     |
| `device`         | 执行推理的设备。如果为`None`，推理器将选择最合适的一个。   |
| `scope`          | 定义模型模块的名称空间                                     |

推理器设计用于处理预测的可视化和保存。下面是使用`MMPoseInferencer`<mark>执行推理</mark>时可用的参数列表：

| Argument            | Description                                                                                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `show`              | 确定图像或视频的预测结果是否应在弹出窗口中显示。                                                                                                                              |
| `radius`            | 设置关键点半径。                                                                                                                                                              |
| `thickness`         | 设置骨架（线条）粗细。                                                                                                                                                        |
| `return_vis`        | 确定返回结果`result`中是否应包括可视化结果列表`visualization`。                                                                                                               |
| `vis_out_dir`       | 指定保存可视化图像的文件夹路径。如果未设置，将不会保存可视化图像。                                                                                                            |
| `return_datasample` | 确定是否以`PoseDataSample`的形式返回预测。                                                                                                                                    |
| `pred_out_dir`      | 指定保存预测结果`predictions`的文件夹路径。如果不设置，预测结果将不会被保存。                                                                                                 |
| `out_dir`           | 如果指定了输出路径参数`out_dir`，但未设置`vis_out_dir`或`pred_out_dir`，则分别将`vis_out_dir`或`pred_out_dir`设置为`f'{out_dir}/visualization'`或` f'{out_dir}/ forecasts'`。 |

### 模型别名

MMPose为常用模型提供了一组预定义的别名。在初始化`MMPoseInferencer`时，这些别名可以用作简略的表达方式，而不是指定完整的模型配置名称。下面是可用的模型别名及其对应的配置名称的列表：

| 别名      | 配置文件名称                                       | 对应任务                        | 姿态估计模型  | 检测模型            |
| --------- | -------------------------------------------------- | ------------------------------- | ------------- | ------------------- |
| animal    | rtmpose-m_8xb64-210e_ap10k-256x256                 | Animal pose estimation          | RTMPose-m     | RTMDet-m            |
| human     | rtmpose-m_8xb256-420e_aic-coco-256x192             | Human pose estimation           | RTMPose-m     | RTMDet-m            |
| face      | rtmpose-m_8xb64-60e_wflw-256x256                   | Face keypoint detection         | RTMPose-m     | yolox-s             |
| hand      | rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256   | Hand keypoint detection         | RTMPose-m     | ssdlite_mobilenetv2 |
| wholebody | rtmpose-m_8xb64-270e_coco-wholebody-256x192        | Human wholebody pose estimation | RTMPose-m     | RTMDet-m            |
| vitpose   | td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-base  | RTMDet-m            |
| vitpose-s | td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192 | Human pose estimation           | ViTPose-small | RTMDet-m            |
| vitpose-b | td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-base  | RTMDet-m            |
| vitpose-l | td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192 | Human pose estimation           | ViTPose-large | RTMDet-m            |
| vitpose-h | td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-huge  | RTMDet-m            |

此外，用户可以使用CLI工具显示所有可用的别名，使用以下命令:

```shell
python demo/inferencer_demo.py --show-alias
```
