<!-- TOC -->

- [FAQ](#faq)
  - [Installation](#installation)
  - [Coding](#coding)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Deployment](#deployment)

<!-- TOC -->

# FAQ

We list some common issues faced by many users and their corresponding solutions here.
Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.
If the contents here do not cover your issue, please create an issue using the [provided templates](/.github/ISSUE_TEMPLATE/error-report.md) and make sure you fill in all required information in the template.

## Installation

- Compatibility issue between MMCV and MMPose; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

  Compatible MMPose and MMCV versions are shown as below. Please choose the correct version of MMCV to avoid installation issues.

| MMPose version |       MMCV version        |
| :------------: | :-----------------------: |
|     master     | mmcv-full>=1.3.8, \<1.7.0 |
|     0.26.0     | mmcv-full>=1.3.8, \<1.6.0 |
|     0.25.1     | mmcv-full>=1.3.8, \<1.6.0 |
|     0.25.0     | mmcv-full>=1.3.8, \<1.5.0 |
|     0.24.0     | mmcv-full>=1.3.8, \<1.5.0 |
|     0.23.0     | mmcv-full>=1.3.8, \<1.5.0 |
|     0.22.0     | mmcv-full>=1.3.8, \<1.5.0 |
|     0.21.0     | mmcv-full>=1.3.8, \<1.5.0 |
|     0.20.0     | mmcv-full>=1.3.8, \<1.4.0 |
|     0.19.0     | mmcv-full>=1.3.8, \<1.4.0 |
|     0.18.0     | mmcv-full>=1.3.8, \<1.4.0 |
|     0.17.0     | mmcv-full>=1.3.8, \<1.4.0 |
|     0.16.0     | mmcv-full>=1.3.8, \<1.4.0 |
|     0.14.0     | mmcv-full>=1.1.3, \<1.4.0 |
|     0.13.0     | mmcv-full>=1.1.3, \<1.4.0 |
|     0.12.0     |  mmcv-full>=1.1.3, \<1.3  |
|     0.11.0     |  mmcv-full>=1.1.3, \<1.3  |
|     0.10.0     |  mmcv-full>=1.1.3, \<1.3  |
|     0.9.0      |  mmcv-full>=1.1.3, \<1.3  |
|     0.8.0      |  mmcv-full>=1.1.1, \<1.2  |
|     0.7.0      |  mmcv-full>=1.1.1, \<1.2  |

- **Unable to install xtcocotools**

  1. Try to install it using pypi manually `pip install xtcocotools`.
  2. If step1 does not work. Try to install it from [source](https://github.com/jin-s13/xtcocoapi).

  ```
  git clone https://github.com/jin-s13/xtcocoapi
  cd xtcocoapi
  python setup.py install
  ```

- **No matching distribution found for xtcocotools>=1.6**

  1. Install cython by `pip install cython`.
  2. Install xtcocotools from [source](https://github.com/jin-s13/xtcocoapi).

  ```
  git clone https://github.com/jin-s13/xtcocoapi
  cd xtcocoapi
  python setup.py install
  ```

- **"No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'"**

  1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`.
  2. Install mmcv-full following the [installation instruction](https://mmcv.readthedocs.io/en/latest/#installation).

## Coding

- Do I need to reinstall mmpose after some code modifications?

  If you follow [the best practice](install.md) and install mmpose from source, any local modifications made to the code will take effect without reinstallation.

- How to develop with multiple `MMPose` versions?

  Generally speaking, we recommend to use different virtual environments to manage `MMPose` in different working directories. However, you can also use the same environment to develop `MMPose` in different folders, like `mmpose-0.26.0`, `mmpose-0.25.0`. When you run the train or test shell script, it will adopt the `mmpose` package in the current folder. And when you run other Python script, you can also add `` PYTHONPATH=`pwd`  `` at the beginning of your command to use the package in the current folder.

  Conversely, to use the default `MMPose` installed in the environment rather than the one you are working with, you can remove the following line in those shell scripts:

  ```shell
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
  ```

## Data

- **How to convert my 2d keypoint dataset to coco-type?**

  You may refer to this conversion [tool](https://github.com/open-mmlab/mmpose/blob/master/tools/dataset/parse_macaquepose_dataset.py) to prepare your data.
  Here is an [example](https://github.com/open-mmlab/mmpose/blob/master/tests/data/macaque/test_macaque.json) of the coco-type json.
  In the coco-type json, we need "categories", "annotations" and "images". "categories" contain some basic information of the dataset, e.g. class name and keypoint names.
  "images" contain image-level information. We need "id", "file_name", "height", "width". Others are optional.
  Note: (1) It is okay that "id"s are not continuous or not sorted (e.g. 1000, 40, 352, 333 ...).

  "annotations" contain instance-level information. We need "image_id", "id", "keypoints", "num_keypoints", "bbox", "iscrowd", "area", "category_id". Others are optional.
  Note: (1) "num_keypoints" means the number of visible keypoints. (2) By default, please set "iscrowd: 0". (3) "area" can be calculated using the bbox (area = w * h) (4) Simply set "category_id: 1". (5) The "image_id" in "annotations" should match the "id" in "images".

- **What if my custom dataset does not have bounding box label?**

  We can estimate the bounding box of a person as the minimal box that tightly bounds all the keypoints.

- **What if my custom dataset does not have segmentation label?**

  Just set the `area` of the person as the area of the bounding boxes. During evaluation, please set `use_area=False` as in this [example](https://github.com/open-mmlab/mmpose/blob/a82dd486853a8a471522ac06b8b9356db61f8547/mmpose/datasets/datasets/top_down/topdown_aic_dataset.py#L113).

- **What is `COCO_val2017_detections_AP_H_56_person.json`? Can I train pose models without it?**

  "COCO_val2017_detections_AP_H_56_person.json" contains the "detected" human bounding boxes for COCO validation set, which are generated by FasterRCNN.
  One can choose to use gt bounding boxes to evaluate models, by setting `use_gt_bbox=True` and `bbox_file=''`. Or one can use detected boxes to evaluate
  the generalizability of models, by setting `use_gt_bbox=False` and `bbox_file='COCO_val2017_detections_AP_H_56_person.json'`.

## Training

- **RuntimeError: Address already in use**

  Set the environment variables `MASTER_PORT=XXX`. For example,
  `MASTER_PORT=29517 GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=2 ./tools/slurm_train.sh Test res50 configs/body/2D_Kpt_SV_RGB_Img/topdown_hm/coco/res50_coco_256x192.py work_dirs/res50_coco_256x192`

- **"Unexpected keys in source state dict" when loading pre-trained weights**

  It's normal that some layers in the pretrained model are not used in the pose model. ImageNet-pretrained classification network and the pose network may have different architectures (e.g. no classification head). So some unexpected keys in source state dict is actually expected.

- **How to use trained models for backbone pre-training ?**

  Refer to [Use Pre-Trained Model](/docs/en/tutorials/1_finetune.md#use-pre-trained-model),
  in order to use the pre-trained model for the whole network (backbone + head), the new config adds the link of pre-trained models in the `load_from`.

  And to use backbone for pre-training, you can change `pretrained` value in the backbone dict of config files to the checkpoint path / url.
  When training, the unexpected keys will be ignored.

- **How to visualize the training accuracy/loss curves in real-time ?**

  Use `TensorboardLoggerHook` in `log_config` like

  ```python
  log_config=dict(interval=20, hooks=[dict(type='TensorboardLoggerHook')])
  ```

  You can refer to [customize_runtime.md](/docs/en/tutorials/6_customize_runtime.md#log-config) and the example [config](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L26).

- **Log info is NOT printed**

  Use smaller log interval. For example, change `interval=50` to `interval=1` in the [config](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L23).

- **How to fix stages of backbone when finetuning a model ?**

  You can refer to [`def _freeze_stages()`](https://github.com/open-mmlab/mmpose/blob/d026725554f9dc08e8708bd9da8678f794a7c9a6/mmpose/models/backbones/resnet.py#L618) and [`frozen_stages`](https://github.com/open-mmlab/mmpose/blob/d026725554f9dc08e8708bd9da8678f794a7c9a6/mmpose/models/backbones/resnet.py#L498),
  reminding to set `find_unused_parameters = True` in config files for distributed training or testing.

## Evaluation

- **How to evaluate on MPII test dataset?**

  Since we do not have the ground-truth for test dataset, we cannot evaluate it 'locally'.
  If you would like to evaluate the performance on test set, you have to upload the `pred.mat` (which is generated during testing) to the official server via email, according to [the MPII guideline](http://human-pose.mpi-inf.mpg.de/#evaluation).

- **For top-down 2d pose estimation, why predicted joint coordinates can be out of the bounding box (bbox)?**

  We do not directly use the bbox to crop the image. Bbox will be first transformed to center & scale, and the scale will be multiplied by a factor (1.25) to include some context. If the ratio of width/height is different from that of model input (possibly 192/256), we will adjust the bbox. You can refer to [the code](https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/pipelines/top_down_transform.py#L15) for more details.

## Inference

- **How to run mmpose on CPU?**

  Run demos with `--device=cpu`.

- **How to speed up inference?**

  For top-down models, try to edit the config file. For example,

  1. set `flip_test=False` in [topdown-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L51).
  2. set `post_process='default'` in [topdown-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L52).
  3. use faster human bounding box detector, see [MMDetection](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).

  For bottom-up models, try to edit the config file. For example,

  1. set `flip_test=False` in [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L91).
  2. set `adjust=False` in [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L89).
  3. set `refine=False` in [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L90).
  4. use smaller input image size in [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L39).

## Deployment

- **Why is the onnx model converted by mmpose throwing error when converting to other frameworks such as TensorRT?**

  For now, we can only make sure that models in mmpose are onnx-compatible. However, some operations in onnx may be unsupported by your target framework for deployment, e.g. TensorRT in [this issue](https://github.com/open-mmlab/mmaction2/issues/414).

  And please note that `pytorch2onnx` in `MMPose` is no longer maintained and will be deprecated in the future. We have [MMDeploy](https://github.com/open-mmlab/mmdeploy) to support model deployment for all `OpenMMLab` codebases including `MMPose`. You can find details about supported models and user guides in their [documentation](https://mmdeploy.readthedocs.io/en/latest/), and raise issues to request support for the models you would like to use.
