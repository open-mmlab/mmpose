# 准备数据集

在这份文档将指导如何为 MMPose 准备数据集，包括使用内置数据集、创建自定义数据集、结合数据集进行训练、浏览和下载数据集。

## 使用内置数据集

**步骤一**: 准备数据

MMPose 支持多种任务和相应的数据集。你可以在 [数据集仓库](https://mmpose.readthedocs.io/en/latest/dataset_zoo.html) 中找到它们。为了正确准备你的数据，请按照你选择的数据集的指南进行操作。

**步骤二**: 在配置文件中进行数据集设置

在开始训练或评估模型之前，你必须配置数据集设置。以 [`td-hm_hrnet-w32_8xb64-210e_coco-256x192.py`](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) 为例，它可以用于在 COCO 数据集上训练或评估 HRNet 姿态估计器。下面我们浏览一下数据集配置：

- 基础数据集参数

  ```python
  # base dataset settings
  dataset_type = 'CocoDataset'
  data_mode = 'topdown'
  data_root = 'data/coco/'
  ```

  - `dataset_type` 指定数据集的类名。用户可以参考 [数据集 API](https://mmpose.readthedocs.io/en/latest/api.html#datasets) 来找到他们想要的数据集的类名。
  - `data_mode` 决定了数据集的输出格式，有两个选项可用：`'topdown'` 和 `'bottomup'`。如果 `data_mode='topdown'`，数据元素表示一个实例及其姿态；否则，一个数据元素代表一张图像，包含多个实例和姿态。
  - `data_root` 指定数据集的根目录。

- 数据处理流程

  ```python
  # pipelines
  train_pipeline = [
      dict(type='LoadImage'),
      dict(type='GetBBoxCenterScale'),
      dict(type='RandomFlip', direction='horizontal'),
      dict(type='RandomHalfBody'),
      dict(type='RandomBBoxTransform'),
      dict(type='TopdownAffine', input_size=codec['input_size']),
      dict(type='GenerateTarget', encoder=codec),
      dict(type='PackPoseInputs')
  ]
  val_pipeline = [
      dict(type='LoadImage'),
      dict(type='GetBBoxCenterScale'),
      dict(type='TopdownAffine', input_size=codec['input_size']),
      dict(type='PackPoseInputs')
  ]
  ```

  `train_pipeline` 和 `val_pipeline` 分别定义了训练和评估阶段处理数据元素的步骤。除了加载图像和打包输入之外，`train_pipeline` 主要包含数据增强技术和目标生成器，而 `val_pipeline` 则专注于将数据元素转换为统一的格式。

- 数据加载器

  ```python
  # data loaders
  train_dataloader = dict(
      batch_size=64,
      num_workers=2,
      persistent_workers=True,
      sampler=dict(type='DefaultSampler', shuffle=True),
      dataset=dict(
          type=dataset_type,
          data_root=data_root,
          data_mode=data_mode,
          ann_file='annotations/person_keypoints_train2017.json',
          data_prefix=dict(img='train2017/'),
          pipeline=train_pipeline,
      ))
  val_dataloader = dict(
      batch_size=32,
      num_workers=2,
      persistent_workers=True,
      drop_last=False,
      sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
      dataset=dict(
          type=dataset_type,
          data_root=data_root,
          data_mode=data_mode,
          ann_file='annotations/person_keypoints_val2017.json',
          bbox_file='data/coco/person_detection_results/'
          'COCO_val2017_detections_AP_H_56_person.json',
          data_prefix=dict(img='val2017/'),
          test_mode=True,
          pipeline=val_pipeline,
      ))
  test_dataloader = val_dataloader
  ```

  这个部分是配置数据集的关键。除了前面讨论过的基础数据集参数和数据处理流程之外，这里还定义了其他重要的参数。`batch_size` 决定了每个 GPU 的 batch size；`ann_file` 指定了数据集的注释文件；`data_prefix` 指定了图像文件夹。`bbox_file` 仅在 top-down 数据集的 val/test 数据加载器中使用，用于提供检测到的边界框信息。

我们推荐从使用相同数据集的配置文件中复制数据集配置，而不是从头开始编写，以最小化潜在的错误。通过这样做，用户可以根据需要进行必要的修改，从而确保更可靠和高效的设置过程。

## 使用自定义数据集

[自定义数据集](../advanced_guides/customize_datasets.md) 指南提供了如何构建自定义数据集的详细信息。在本节中，我们将强调一些使用和配置自定义数据集的关键技巧。

- 确定数据集类名。如果你将数据集重组为 COCO 格式，你可以简单地使用 `CocoDataset` 作为 `dataset_type` 的值。否则，你将需要使用你添加的自定义数据集类的名称。

- 指定元信息配置文件。MMPose 1.x 采用了与 MMPose 0.x 不同的策略来指定元信息。在 MMPose 1.x 中，用户可以按照以下方式指定元信息配置文件：

  ```python
  train_dataloader = dict(
      ...
      dataset=dict(
          type=dataset_type,
          data_root='root/of/your/train/data',
          ann_file='path/to/your/train/json',
          data_prefix=dict(img='path/to/your/train/img'),
          # specify dataset meta information
          metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
          ...),
  )
  ```

  注意，`metainfo` 参数必须在 val/test 数据加载器中指定。

## 使用混合数据集进行训练

MMPose 提供了一个方便且多功能的解决方案，用于训练混合数据集。请参考[混合数据集训练](./mixed_datasets.md)。

## 浏览数据集

`tools/analysis_tools/browse_dataset.py` 帮助用户可视化地浏览姿态数据集，或将图像保存到指定的目录。

```shell
python tools/misc/browse_dataset.py ${CONFIG} [-h] [--output-dir ${OUTPUT_DIR}] [--not-show] [--phase ${PHASE}] [--mode ${MODE}] [--show-interval ${SHOW_INTERVAL}]
```

| ARGS                             | Description                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `CONFIG`                         | 配置文件的路径                                                                                             |
| `--output-dir OUTPUT_DIR`        | 保存可视化结果的目标文件夹。如果不指定，可视化的结果将不会被保存                                           |
| `--not-show`                     | 不适用外部窗口显示可视化的结果                                                                             |
| `--phase {train, val, test}`     | 数据集选项                                                                                                 |
| `--mode {original, transformed}` | 指定可视化图片类型。 `original` 为不使用数据增强的原始图片及标注可视化; `transformed` 为经过增强后的可视化 |
| `--show-interval SHOW_INTERVAL`  | 显示图片的时间间隔                                                                                         |

例如，用户想要可视化 COCO 数据集中的图像和标注，可以使用：

```shell
python tools/misc/browse_dataset.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-e210_coco-256x192.py --mode original
```

检测框和关键点将被绘制在原始图像上。下面是一个例子：
![original_coco](https://user-images.githubusercontent.com/26127467/187383698-7e518f21-b4cc-4712-9e97-99ddd8f0e437.jpg)

原始图像在被输入模型之前需要被处理。为了可视化预处理后的图像和标注，用户需要将参数 `mode` 修改为 `transformed`。例如：

```shell
python tools/misc/browse_dataset.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-e210_coco-256x192.py --mode transformed
```

这是一个处理后的样本：

![transformed_coco](https://user-images.githubusercontent.com/26127467/187386652-bd47335d-797c-4e8c-b823-2a4915f9812f.jpg)

热图目标将与之一起可视化，如果它是在 pipeline 中生成的。

## 用 MIM 下载数据集

通过使用 [OpenDataLab](https://opendatalab.com/)，您可以直接下载开源数据集。通过平台的搜索功能，您可以快速轻松地找到他们正在寻找的数据集。使用平台上的格式化数据集，您可以高效地跨数据集执行任务。

如果您使用 MIM 下载，请确保版本大于 v0.3.8。您可以使用以下命令进行更新、安装、登录和数据集下载：

```shell
# upgrade your MIM
pip install -U openmim

# install OpenDataLab CLI tools
pip install -U opendatalab
# log in OpenDataLab, registry
odl login

# download coco2017 and preprocess by MIM
mim download mmpose --dataset coco2017
```

### 已支持的数据集

下面是支持的数据集列表，更多数据集将在之后持续更新：

#### 人体数据集

| Dataset name  | Download command                          |
| ------------- | ----------------------------------------- |
| COCO 2017     | `mim download mmpose --dataset coco2017`  |
| MPII          | `mim download mmpose --dataset mpii`      |
| AI Challenger | `mim download mmpose --dataset aic`       |
| CrowdPose     | `mim download mmpose --dataset crowdpose` |

#### 人脸数据集

| Dataset name | Download command                     |
| ------------ | ------------------------------------ |
| LaPa         | `mim download mmpose --dataset lapa` |
| 300W         | `mim download mmpose --dataset 300w` |
| WFLW         | `mim download mmpose --dataset wflw` |

#### 手部数据集

| Dataset name | Download command                           |
| ------------ | ------------------------------------------ |
| OneHand10K   | `mim download mmpose --dataset onehand10k` |
| FreiHand     | `mim download mmpose --dataset freihand`   |
| HaGRID       | `mim download mmpose --dataset hagrid`     |

#### 全身数据集

| Dataset name | Download command                      |
| ------------ | ------------------------------------- |
| Halpe        | `mim download mmpose --dataset halpe` |

#### 动物数据集

| Dataset name | Download command                      |
| ------------ | ------------------------------------- |
| AP-10K       | `mim download mmpose --dataset ap10k` |

#### 服装数据集

Coming Soon
