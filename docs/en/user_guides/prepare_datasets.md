# Prepare Datasets

In this document, we will give a guide on the process of preparing datasets for the MMPose. Various aspects of dataset preparation will be discussed, including using built-in datasets, creating custom datasets, combining datasets for training, browsing and downloading the datasets.

## Use built-in datasets

**Step 1**: Prepare Data

MMPose supports multiple tasks and corresponding datasets. You can find them in [dataset zoo](https://mmpose.readthedocs.io/en/latest/dataset_zoo.html). To properly prepare your data, please follow the guidelines associated with your chosen dataset.

**Step 2**: Configure Dataset Settings in the Config File

Before training or evaluating models, you must configure the dataset settings. Take [`td-hm_hrnet-w32_8xb64-210e_coco-256x192.py`](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) for example, which can be used to train or evaluate the HRNet pose estimator on COCO dataset. We will go through the dataset configuration.

- Basic Dataset Arguments

  ```python
  # base dataset settings
  dataset_type = 'CocoDataset'
  data_mode = 'topdown'
  data_root = 'data/coco/'
  ```

  - `dataset_type` specifies the class name of the dataset. Users can refer to [Datasets APIs](https://mmpose.readthedocs.io/en/latest/api.html#datasets) to find the class name of their desired dataset.
  - `data_mode` determines the output format of the dataset, with two options available: `'topdown'` and `'bottomup'`. If `data_mode='topdown'`, the data element represents a single instance with its pose; otherwise, the data element is an entire image containing multiple instances and poses.
  - `data_root` designates the root directory of the dataset.

- Data Processing Pipelines

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

  The `train_pipeline` and `val_pipeline` define the steps to process data elements during the training and evaluation phases, respectively. In addition to loading images and packing inputs, the `train_pipeline` primarily consists of data augmentation techniques and target generator, while the `val_pipeline` focuses on transforming data elements into a unified format.

- Data Loaders

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

  This section is crucial for configuring the dataset in the config file. In addition to the basic dataset arguments and pipelines discussed earlier, other important parameters are defined here. The `batch_size` determines the batch size per GPU; the `ann_file` indicates the annotation file for the dataset; and `data_prefix` specifies the image folder. The `bbox_file`, which supplies detected bounding box information, is only used in the val/test data loader for top-down datasets.

We recommend copying the dataset configuration from provided config files that use the same dataset, rather than writing it from scratch, in order to minimize potential errors. By doing so, users can simply make the necessary modifications as needed, ensuring a more reliable and efficient setup process.

## Use a custom dataset

The [Customize Datasets](../advanced_guides/customize_datasets.md) guide provides detailed information on how to build a custom dataset. In this section, we will highlight some key tips for using and configuring custom datasets.

- Determine the dataset class name. If you reorganize your dataset into the COCO format, you can simply use `CocoDataset` as the value for `dataset_type`. Otherwise, you will need to use the name of the custom dataset class you added.

- Specify the meta information config file. MMPose 1.x employs a different strategy for specifying meta information compared to MMPose 0.x. In MMPose 1.x, users can specify the meta information config file as follows:

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

  Note that the argument `metainfo` must be specified in the val/test data loaders as well.

## Use mixed datasets for training

MMPose offers a convenient and versatile solution for training with mixed datasets. Please refer to [Use Mixed Datasets for Training](./mixed_datasets.md).

## Browse dataset

`tools/analysis_tools/browse_dataset.py` helps the user to browse a pose dataset visually, or save the image to a designated directory.

```shell
python tools/misc/browse_dataset.py ${CONFIG} [-h] [--output-dir ${OUTPUT_DIR}] [--not-show] [--phase ${PHASE}] [--mode ${MODE}] [--show-interval ${SHOW_INTERVAL}]
```

| ARGS                             | Description                                                                                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG`                         | The path to the config file.                                                                                                                         |
| `--output-dir OUTPUT_DIR`        | The target folder to save visualization results. If not specified, the visualization results will not be saved.                                      |
| `--not-show`                     | Do not show the visualization results in an external window.                                                                                         |
| `--phase {train, val, test}`     | Options for dataset.                                                                                                                                 |
| `--mode {original, transformed}` | Specify the type of visualized images. `original` means to show images without pre-processing; `transformed` means to show images are pre-processed. |
| `--show-interval SHOW_INTERVAL`  | Time interval between visualizing two images.                                                                                                        |

For instance, users who want to visualize images and annotations in COCO dataset use:

```shell
python tools/misc/browse_dataset.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-e210_coco-256x192.py --mode original
```

The bounding boxes and keypoints will be plotted on the original image. Following is an example:
![original_coco](https://user-images.githubusercontent.com/26127467/187383698-7e518f21-b4cc-4712-9e97-99ddd8f0e437.jpg)

The original images need to be processed before being fed into models. To visualize pre-processed images and annotations, users need to modify the argument `mode`  to `transformed`. For example:

```shell
python tools/misc/browse_dataset.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-e210_coco-256x192.py --mode transformed
```

Here is a processed sample

![transformed_coco](https://user-images.githubusercontent.com/26127467/187386652-bd47335d-797c-4e8c-b823-2a4915f9812f.jpg)

The heatmap target will be visualized together if it is generated in the pipeline.

## Download dataset via MIM

By using [OpenDataLab](https://opendatalab.com/), you can obtain free formatted datasets in various fields. Through the search function of the platform, you may address the dataset they look for quickly and easily. Using the formatted datasets from the platform, you can efficiently conduct tasks across datasets.

If you use MIM to download, make sure that the version is greater than v0.3.8. You can use the following command to update, install, login and download the dataset:

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

### Supported datasets

Here is the list of supported datasets, we will continue to update it in the future.

#### Body

| Dataset name  | Download command                          |
| ------------- | ----------------------------------------- |
| COCO 2017     | `mim download mmpose --dataset coco2017`  |
| MPII          | `mim download mmpose --dataset mpii`      |
| AI Challenger | `mim download mmpose --dataset aic`       |
| CrowdPose     | `mim download mmpose --dataset crowdpose` |

#### Face

| Dataset name | Download command                     |
| ------------ | ------------------------------------ |
| LaPa         | `mim download mmpose --dataset lapa` |
| 300W         | `mim download mmpose --dataset 300w` |
| WFLW         | `mim download mmpose --dataset wflw` |

#### Hand

| Dataset name | Download command                           |
| ------------ | ------------------------------------------ |
| OneHand10K   | `mim download mmpose --dataset onehand10k` |
| FreiHand     | `mim download mmpose --dataset freihand`   |
| HaGRID       | `mim download mmpose --dataset hagrid`     |

#### Whole Body

| Dataset name | Download command                      |
| ------------ | ------------------------------------- |
| Halpe        | `mim download mmpose --dataset halpe` |

#### Animal

| Dataset name | Download command                      |
| ------------ | ------------------------------------- |
| AP-10K       | `mim download mmpose --dataset ap10k` |

#### Fashion

Coming Soon
