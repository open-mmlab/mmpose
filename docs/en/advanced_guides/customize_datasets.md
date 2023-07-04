# Customize Datasets

## Customize datasets by reorganizing data to COCO format

The simplest way to use the custom dataset is to convert your annotation format to COCO dataset format.

The annotation JSON files in COCO format have the following necessary keys:

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

There are three necessary keys in the json file:

- `images`: contains a list of images with their information like `file_name`, `height`, `width`, and `id`.
- `annotations`: contains the list of instance annotations.
- `categories`: contains the category name ('person') and its ID (1).

If the annotations have been organized in COCO format, there is no need to create a new dataset class. You can use `CocoDataset` class alternatively.

## Create a custom dataset_info config file for the dataset

Add a new dataset info config file that contains the metainfo about the dataset.

```
configs/_base_/datasets/custom.py
```

An example of the dataset config is as follows.

`keypoint_info` contains the information about each keypoint.

1. `name`: the keypoint name. The keypoint name must be unique.
2. `id`: the keypoint id.
3. `color`: (\[B, G, R\]) is used for keypoint visualization.
4. `type`: 'upper' or 'lower', will be used in data augmentation.
5. `swap`: indicates the 'swap pair' (also known as 'flip pair'). When applying image horizontal flip, the left part will become the right part. We need to flip the keypoints accordingly.

`skeleton_info` contains information about the keypoint connectivity, which is used for visualization.

`joint_weights` assigns different loss weights to different keypoints.

`sigmas` is used to calculate the OKS score. You can read [keypoints-eval](https://cocodataset.org/#keypoints-eval) to learn more about it.

Here is an simplified example of dataset_info config file ([full text](/configs/_base_/datasets/coco.py)).

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

## Create a custom dataset class

If the annotations are not organized in COCO format, you need to create a custom dataset class by the following steps:

1. First create a package inside the `mmpose/datasets/datasets` folder.

2. Create a class definition of your dataset in the package folder and register it in the registry with a name. Without a name, it will keep giving the error. `KeyError: 'XXXXX is not in the dataset registry'`

   ```
   from mmengine.dataset import BaseDataset
   from mmpose.registry import DATASETS

   @DATASETS.register_module(name='MyCustomDataset')
   class MyCustomDataset(BaseDataset):
   ```

   You can refer to [this doc](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html) on how to build customed dataset class with `mmengine.BaseDataset`.

3. Make sure you have updated the `__init__.py` of your package folder

4. Make sure you have updated the `__init__.py` of the dataset package folder.

## Create a custom training config file

Create a custom training config file as per your need and the model/architecture you want to use in the configs folder. You may modify an existing config file to use the new custom dataset.

In `configs/my_custom_config.py`:

```python
...
# dataset and dataloader settings
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

Make sure you have provided all the paths correctly.

## Dataset Wrappers

The following dataset wrappers are supported in [MMEngine](https://github.com/open-mmlab/mmengine), you can refer to [MMEngine tutorial](https://mmengine.readthedocs.io/en/latest) to learn how to use it.

- [ConcatDataset](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html#concatdataset)
- [RepeatDataset](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html#repeatdataset)

### CombinedDataset

MMPose provides `CombinedDataset` to combine multiple datasets with different annotations. A combined dataset can be defined in config files as:

```python
dataset_1 = dict(
    type='dataset_type_1',
    data_root='root/of/your/dataset1',
    data_prefix=dict(img_path='path/to/your/img'),
    ann_file='annotations/train.json',
    pipeline=[
        # the converter transforms convert data into a unified format
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

- **MetaInfo of combined dataset** determines the annotation format. Either metainfo of a sub-dataset or a customed dataset metainfo is valid here. To custom a dataset metainfo, please refer to [Create a custom dataset_info config file for the dataset](#create-a-custom-datasetinfo-config-file-for-the-dataset).

- **Converter transforms of sub-datasets** are applied when there exist mismatches of annotation format between sub-datasets and the combined dataset. For example, the number and order of keypoints might be different in the combined dataset and the sub-datasets. Then `KeypointConverter` can be used to unify the keypoints number and order.

- More details about `CombinedDataset` and `KeypointConverter` can be found in Advanced Guides-[Training with Mixed Datasets](../user_guides/mixed_datasets.md).
