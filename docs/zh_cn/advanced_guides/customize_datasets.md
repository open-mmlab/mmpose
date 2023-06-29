# 自定义数据集

MMPose 目前已支持了多个任务和相应的数据集。您可以在 [数据集](https://mmpose.readthedocs.io/zh_CN/latest/dataset_zoo.html) 找到它们。请按照相应的指南准备数据。

<!-- TOC -->

- [自定义数据集-将数据组织为 COCO 格式](#自定义数据集-将数据组织为-coco-格式)
- [创建自定义数据集的元信息文件](#创建自定义数据集的元信息文件)
- [创建自定义数据集类](#创建自定义数据集类)
- [创建自定义配置文件](#创建自定义配置文件)
- [数据集封装](#数据集封装)

<!-- TOC -->

## 将数据组织为 COCO 格式

最简单的使用自定义数据集的方法是将您的注释格式转换为 COCO 数据集格式。

COCO 格式的注释 JSON 文件具有以下必要键：

```python
'images': [
    {
        'file_name': '000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
],
'annotations': [
    {
        'segmentation': [[426.36,
            ...
            424.34,
            223.3]],
        'keypoints': [0,0,0,
            0,0,0,
            0,0,0,
            427,220,2,
            443,222,2,
            414,228,2,
            449,232,2,
            408,248,1,
            454,261,2,
            0,0,0,
            0,0,0,
            411,287,2,
            431,287,2,
            0,0,0,
            458,265,2,
            0,0,0,
            466,300,1],
        'num_keypoints': 10,
        'area': 3894.5826,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [402.34, 205.02, 65.26, 88.45],
        'category_id': 1,
        'id': 215218
    },
    ...
],
'categories': [
    {'id': 1, 'name': 'person'},
 ]
```

JSON 标注文件中有三个关键词是必需的：

- `images`：包含所有图像信息的列表，每个图像都有一个 `file_name`、`height`、`width` 和 `id` 键。
- `annotations`：包含所有实例标注信息的列表，每个实例都有一个 `segmentation`、`keypoints`、`num_keypoints`、`area`、`iscrowd`、`image_id`、`bbox`、`category_id` 和 `id` 键。
- `categories`：包含所有类别信息的列表，每个类别都有一个 `id` 和 `name` 键。以人体姿态估计为例，`id` 为 1，`name` 为 `person`。

如果您的数据集已经是 COCO 格式的，那么您可以直接使用 `CocoDataset` 类来读取该数据集。

## 创建自定义数据集的元信息文件

对于一个新的数据集而言，您需要创建一个新的数据集元信息文件。该文件包含了数据集的基本信息，如关键点个数、排列顺序、可视化颜色、骨架连接关系等。元信息文件通常存放在 `config/_base_/datasets/` 目录下，例如：

```
config/_base_/datasets/custom.py
```

元信息文件中需要包含以下信息：

- `keypoint_info`：每个关键点的信息：
  1. `name`: 关键点名称，必须是唯一的，例如 `nose`、`left_eye` 等。
  2. `id`: 关键点 ID，必须是唯一的，从 0 开始。
  3. `color`: 关键点可视化时的颜色，以 (\[B, G, R\]) 格式组织起来，用于可视化。
  4. `type`: 关键点类型，可以是 `upper`、`lower` 或 \`\`，用于数据增强。
  5. `swap`: 关键点交换关系，用于水平翻转数据增强。
- `skeleton_info`：骨架连接关系，用于可视化。
- `joint_weights`：每个关键点的权重，用于损失函数计算。
- `sigma`：标准差，用于计算 OKS 分数，详细信息请参考 [keypoints-eval](https://cocodataset.org/#keypoints-eval)。

下面是一个简化版本的元信息文件（[完整版](/configs/_base_/datasets/coco.py)）：

```python
dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        ...
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        ...
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
```

## 创建自定义数据集类

如果标注信息不是用 COCO 格式存储的，那么您需要创建一个新的数据集类。数据集类需要继承自 `BaseDataset` 类，并且需要按照以下步骤实现：

1. 在 `mmpose/datasets/datasets` 目录下找到该数据集符合的 package，如果没有符合的，则创建一个新的 package。

2. 在该 package 下创建一个新的数据集类，在对应的注册器中进行注册：

   ```python
   from mmengine.dataset import BaseDataset
   from mmpose.registry import DATASETS

   @DATASETS.register_module(name='MyCustomDataset')
   class MyCustomDataset(BaseDataset):
   ```

   如果未注册，你会在运行时遇到 `KeyError: 'XXXXX is not in the dataset registry'`。
   关于 `mmengine.BaseDataset` 的更多信息，请参考 [这个文档](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html)。

3. 确保你在 package 的 `__init__.py` 中导入了该数据集类。

4. 确保你在 `mmpose/datasets/__init__.py` 中导入了该 package。

## 创建自定义配置文件

在配置文件中，你需要修改跟数据集有关的部分，例如：

```python
...
# 自定义数据集类
dataset_type = 'MyCustomDataset' # or 'CocoDataset'

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root='root/of/your/train/data',
        ann_file='path/to/your/train/json',
        data_prefix=dict(img='path/to/your/train/img'),
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
        ...),
    )

val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root='root/of/your/val/data',
        ann_file='path/to/your/val/json',
        data_prefix=dict(img='path/to/your/val/img'),
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
        ...),
    )

test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root='root/of/your/test/data',
        ann_file='path/to/your/test/json',
        data_prefix=dict(img='path/to/your/test/img'),
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
        ...),
    )
...
```

请确保所有的路径都是正确的。

## 数据集封装

目前 [MMEngine](https://github.com/open-mmlab/mmengine) 支持以下数据集封装：

- [ConcatDataset](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/basedataset.html#concatdataset)
- [RepeatDataset](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/basedataset.html#repeatdataset)

### CombinedDataset

MMPose 提供了一个 `CombinedDataset` 类，它可以将多个数据集封装成一个数据集。它的使用方法如下：

```python
dataset_1 = dict(
    type='dataset_type_1',
    data_root='root/of/your/dataset1',
    data_prefix=dict(img_path='path/to/your/img'),
    ann_file='annotations/train.json',
    pipeline=[
        # 使用转换器将标注信息统一为需要的格式
        converter_transform_1
    ])

dataset_2 = dict(
    type='dataset_type_2',
    data_root='root/of/your/dataset2',
    data_prefix=dict(img_path='path/to/your/img'),
    ann_file='annotations/train.json',
    pipeline=[
        converter_transform_2
    ])

shared_pipeline = [
    LoadImage(),
    ParseImage(),
]

combined_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='path/to/your/metainfo'),
    datasets=[dataset_1, dataset_2],
    pipeline=shared_pipeline,
)
```

- **合并数据集的元信息** 决定了标注格式，可以是子数据集的元信息，也可以是自定义的元信息。如果要自定义元信息，可以参考 [创建自定义数据集的元信息文件](#创建自定义数据集的元信息文件)。
- **KeypointConverter** 用于将不同的标注格式转换成统一的格式。比如将关键点个数不同、关键点排列顺序不同的数据集进行合并。
- 更详细的说明请前往[混合数据集训练](../user_guides/mixed_datasets.md)。
