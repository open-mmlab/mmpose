# 教程 2: 增加新的数据集

## 通过将数据组织为已有格式来添加自定义数据集

使用自定义数据集最简单的方法是将其转换为现有的COCO数据集格式。

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

在数据预处理完成后，用户需要修改配置文件以使用该数据集。

在 `configs/my_custom_config.py` 文件中，需要进行如下修改：

```python
...
# 数据集设定
dataset_type = 'MyCustomDataset'
classes = ('a', 'b', 'c', 'd', 'e')
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
