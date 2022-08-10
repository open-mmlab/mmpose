# 教程 2: 增加新的数据集

## 将数据集转化为COCO格式

我们首先需要将自定义数据集，转换为COCO数据集格式。

COCO数据集格式的json标注文件有以下关键字：

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

Json文件中必须包含以下三个关键字：

- `images`： 包含图片信息的列表，提供图片的 `file_name`， `height`， `width` 和 `id` 等信息。
- `annotations`： 包含实例标注的列表。
- `categories`： 包含类别名称 （'person'） 和对应的 ID (1)。

## 为自定义数据集创建 dataset_info 数据集配置文件

在如下位置，添加一个数据集配置文件。

```
configs/_base_/datasets/custom.py
```

数据集配置文件的样例如下：

`keypoint_info` 包含每个关键点的信息，其中：

1. `name`: 代表关键点的名称。一个数据集的每个关键点，名称必须唯一。
2. `id`: 关键点的标识号。
3. `color`: (\[B, G, R\]) 用于可视化关键点。
4. `type`: 分为 'upper' 和 'lower' 两种，用于数据增强。
5. `swap`: 表示与当前关键点，“镜像对称”的关键点名称。

`skeleton_info` 包含关键点之间的连接关系，主要用于可视化。

`joint_weights` 可以为不同的关键点设置不同的损失权重，用于训练。

`sigmas` 用于计算 OKS 得分，具体内容请参考 [keypoints-eval](https://cocodataset.org/#keypoints-eval)。

```
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
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
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
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
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

1. 首先在 mmpose/datasets/datasets 文件夹创建一个包，比如命名为 custom。

2. 定义数据集类，并且注册这个类。

   ```
   @DATASETS.register_module(name='MyCustomDataset')
   class MyCustomDataset(SomeOtherBaseClassAsPerYourNeed):
   ```

3. 为你的自定义类别创建 `mmpose/datasets/datasets/custom/__init__.py`

4. 更新 `mmpose/datasets/__init__.py`

## 创建和修改训练配置文件

创建和修改训练配置文件，来使用你的自定义数据集。

在 `configs/my_custom_config.py` 中，修改如下几行。

```python
...
# dataset settings
dataset_type = 'MyCustomDataset'
...
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='path/to/your/train/json',
        img_prefix='path/to/your/train/img',
        ...),
    val=dict(
        type=dataset_type,
        ann_file='path/to/your/val/json',
        img_prefix='path/to/your/val/img',
        ...),
    test=dict(
        type=dataset_type,
        ann_file='path/to/your/test/json',
        img_prefix='path/to/your/test/img',
        ...))
...
```
