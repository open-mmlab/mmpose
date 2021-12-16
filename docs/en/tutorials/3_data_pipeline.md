# Tutorial 3: Custom Data Pipelines

## Design of Data pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading
with multiple workers. `Dataset` returns a dict of data items corresponding
the arguments of models' forward method.
Since the data in pose estimation may not be the same size (image size, gt bbox size, etc.),
we introduce a new `DataContainer` type in MMCV to help collect and distribute
data of different size.
See [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) for more details.

The data preparation pipeline and the dataset is decomposed. Usually a dataset
defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.
A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

The operations are categorized into data loading, pre-processing, formatting, label generating.

Here is an pipeline example for Simple Baseline (ResNet50).

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

For each operation, we list the related dict fields that are added/updated/removed.

### Data loading

`LoadImageFromFile`

- add: img, img_file

### Pre-processing

`TopDownRandomFlip`

- update: img, joints_3d, joints_3d_visible, center

`TopDownHalfBodyTransform`

- update: center, scale

`TopDownGetRandomScaleRotation`

- update: scale, rotation

`TopDownAffine`

- update: img, joints_3d, joints_3d_visible

`NormalizeTensor`

- update: img

### Generating labels

`TopDownGenerateTarget`

- add: target, target_weight

### Formatting

`ToTensor`

- update: 'img'

`Collect`

- add: img_meta (the keys of img_meta is specified by `meta_keys`)
- remove: all other keys except for those specified by `keys`

## Extend and use custom pipelines

1. Write a new pipeline in any file, e.g., `my_pipeline.py`. It takes a dict as input and return a dict.

   ```python
   from mmpose.datasets import PIPELINES

   @PIPELINES.register_module()
   class MyTransform:

      def __call__(self, results):
          results['dummy'] = True
          return results
   ```

1. Import the new class.

   ```python
   from .my_pipeline import MyTransform
   ```

1. Use it in config files.

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
