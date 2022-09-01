# 教程 3: 自定义数据前处理流水线

## 设计数据前处理流水线

参照惯例，MMPose 使用 `Dataset` 和 `DataLoader` 实现多进程数据加载。
`Dataset` 返回一个字典，作为模型的输入。
由于姿态估计任务的数据大小不一定相同（图片大小，边界框大小等），MMPose 使用 MMCV 中的 `DataContainer` 收集和分配不同大小的数据。
详情可见[此处](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py)。

数据前处理流水线和数据集是相互独立的。
通常，数据集定义如何处理标注文件，而数据前处理流水线将原始数据处理成网络输入。
数据前处理流水线包含一系列操作。
每个操作都输入一个字典（dict），新增/更新/删除相关字段，最终输出更新后的字典作为下一个操作的输入。

数据前处理流水线的操作可以被分类为数据加载、预处理、格式化和生成监督等（后文将详细介绍）。

这里以 Simple Baseline (ResNet50) 的数据前处理流水线为例：

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownHalfBodyTransform', num_joints_half_body=8, prob_half_body=0.3),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]
```

下面列出每个操作新增/更新/删除的相关字典字段。

### 数据加载

`LoadImageFromFile`

- 新增： img, img_file

### 预处理

`TopDownRandomFlip`

- 更新： img, joints_3d, joints_3d_visible, center

`TopDownHalfBodyTransform`

- 更新： center, scale

`TopDownGetRandomScaleRotation`

- 更新： scale, rotation

`TopDownAffine`

- 更新： img, joints_3d, joints_3d_visible

`NormalizeTensor`

- 更新： img

### 生成监督

`TopDownGenerateTarget`

- 新增： target, target_weight

### 格式化

`ToTensor`

- 更新： 'img'

`Collect`

- 新增： img_meta (其包含的字段由 `meta_keys` 指定)
- 删除： 除了 `keys` 指定以外的所有字段

## 扩展和使用自定义流水线

1. 将一个新的处理流水线操作写入任一文件中，例如 `my_pipeline.py`。它以一个字典作为输入，并返回一个更新后的字典。

   ```python
   from mmpose.datasets import PIPELINES

   @PIPELINES.register_module()
   class MyTransform:

      def __call__(self, results):
          results['dummy'] = True
          return results
   ```

2. 导入定义好的新类。

   ```python
   from .my_pipeline import MyTransform
   ```

3. 在配置文件中使用它。

   ```python
   train_pipeline = [
   dict(type='LoadImageFromFile'),
   dict(type='TopDownRandomFlip', flip_prob=0.5),
   dict(type='TopDownHalfBodyTransform', num_joints_half_body=8, prob_half_body=0.3),
   dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
   dict(type='TopDownAffine'),
   dict(type='MyTransform'),
   dict(type='ToTensor'),
   dict(
       type='NormalizeTensor',
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]),
   dict(type='TopDownGenerateTarget', sigma=2),
   dict(
       type='Collect',
       keys=['img', 'target', 'target_weight'],
       meta_keys=[
           'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
           'rotation', 'bbox_score', 'flip_pairs'
       ]),
   ]
   ```
