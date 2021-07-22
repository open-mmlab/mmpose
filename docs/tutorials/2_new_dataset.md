# Tutorial 2: Adding New Dataset

## Customize datasets by reorganizing data

### Reorganize dataset to existing format

The simplest way to use the custom dataset is to convert your annotation format to existing COCO dataset format.

The annotation json files in COCO format has the following necessary keys:

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

After the data pre-processing, the users need to further modify the config files to use the dataset.

In `configs/my_custom_config.py`:

```python
...
# dataset settings
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
