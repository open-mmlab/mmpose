# Migration

MMPose 1.0 has made significant BC-breaking changes, with modules redesigned and reorganized to reduce code redundancy and improve efficiency. For developers who have some deep-learning knowledge, this tutorial provides a migration guide.

Whether you are **a user of the previous version of MMPose**, or **a new user wishing to migrate your Pytorch project to MMPose**, you can learn how to build a project based on MMPose 1.0 with this tutorial.

```{note}
This  tutorial covers what developers will concern when using MMPose 1.0:

- Overall code architecture

- How to manage modules with configs

- How to use my own custom datasets

- How to add new modules(backbone, head, loss function, etc.)
```

The content of this tutorial is organized as follows:

- [Migration](#migration)
  - [Overall Code Architecture](#overall-code-architecture)
  - [Step1: Configs](#step1-configs)
  - [Step2: Data](#step2-data)
    - [Dataset Meta Information](#dataset-meta-information)
    - [Dataset](#dataset)
    - [Pipeline](#pipeline)
      - [i. Augmentation](#i-augmentation)
      - [ii. Transformation](#ii-transformation)
      - [iii. Encoding](#iii-encoding)
      - [iv. Packing](#iv-packing)
  - [Step3: Model](#step3-model)
    - [Data Preprocessor](#data-preprocessor)
    - [Backbone](#backbone)
    - [Neck](#neck)
    - [Head](#head)
  - [Compatibility of MMPose 0.X](#compatibility-of-mmpose-0x)
    - [Data Transformation](#data-transformation)
      - [Translation, Rotation and Scaling](#translation-rotation-and-scaling)
      - [Target Generation](#target-generation)
      - [Data Normalization](#data-normalization)
    - [Compatibility of Models](#compatibility-of-models)
      - [Heatmap-based Model](#heatmap-based-model)
      - [RLE-based Model](#rle-based-model)

## Overall Code Architecture

![overall-en](https://user-images.githubusercontent.com/13503330/187372008-2a94bad5-5252-4155-9ae3-3da1c426f569.png)

Generally speaking, there are **five parts** developers will use during project development:

- **General:** Environment, Hook, Checkpoint, Logger, etc.

- **Data:** Dataset, Dataloader, Data Augmentation, etc.

- **Training:** Optimizer, Learning Rate Scheduler, etc.

- **Model:** Backbone, Neck, Head, Loss function, etc.

- **Evaluation:** Metric, Evaluator, etc.

Among them, modules related to **General**, **Training** and **Evaluation** are often provided by the training framework [MMEngine](https://github.com/open-mmlab/mmengine), and developers only need to call APIs and adjust the parameters.  Developers mainly focus on implementing the **Data** and **Model** parts.

## Step1: Configs

In MMPose, we use a Python file as config for the definition and parameter management of the whole project. Therefore, we strongly recommend the developers who use MMPose for the first time to refer to [Configs](./user_guides/configs.md).

Note that all new modules need to be registered using `Registry` and imported in `__init__.py` in the corresponding directory before we can create their instances from configs.

## Step2: Data

The organization of data in MMPose contains:

- Dataset Meta Information

- Dataset

- Pipeline

### Dataset Meta Information

The meta information of a pose dataset usually includes the definition of keypoints and skeleton, symmetrical characteristic, and keypoint properties (e.g. belonging to upper or lower body, weights and sigmas). These information is important in data preprocessing, model training and evaluation. In MMpose, the dataset meta information is stored in configs files under `$MMPOSE/configs/_base_/datasets/`.

To use a custom dataset in MMPose, you need to add a new config file of the dataset meta information. Take the MPII dataset (`$MMPOSE/configs/_base_/datasets/mpii.py`) as an example. Here is its dataset information:

```Python
dataset_info = dict(
    dataset_name='mpii',
    paper_info=dict(
        author='Mykhaylo Andriluka and Leonid Pishchulin and '
        'Peter Gehler and Schiele, Bernt',
        title='2D Human Pose Estimation: New Benchmark and '
        'State of the Art Analysis',
        container='IEEE Conference on Computer Vision and '
        'Pattern Recognition (CVPR)',
        year='2014',
        homepage='http://human-pose.mpi-inf.mpg.de/',
    ),
    keypoint_info={
        0:
        dict(
            name='right_ankle',
            id=0,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        ## omitted
    },
    skeleton_info={
        0:
        dict(link=('right_ankle', 'right_knee'), id=0, color=[255, 128, 0]),
        ## omitted
    },
    joint_weights=[
        1.5, 1.2, 1., 1., 1.2, 1.5, 1., 1., 1., 1., 1.5, 1.2, 1., 1., 1.2, 1.5
    ],
    # Adapted from COCO dataset.
    sigmas=[
        0.089, 0.083, 0.107, 0.107, 0.083, 0.089, 0.026, 0.026, 0.026, 0.026,
        0.062, 0.072, 0.179, 0.179, 0.072, 0.062
    ])
```

### Dataset

To use custom dataset in MMPose, we recommend converting the annotations into a supported format (e.g. COCO or MPII) and directly using our implementation of the corresponding dataset. If this is not applicable, you may need to implement your own dataset class.

Most 2D keypoint datasets in MMPose **organize the annotations in a COCO-like style**. Thus we provide a base class [BaseCocoStyleDataset](mmpose/datasets/datasets/base/base_coco_style_dataset.py) for these datasets. We recommend that users subclass `BaseCocoStyleDataset` and override the methods as needed (usually `__init__()` and `_load_annotations()`) to extend to a new custom 2D keypoint dataset.

```{note}
Please refer to [COCO](./dataset_zoo/2d_body_keypoint.md) for more details about the COCO data format.
```

```{note}
The bbox format in MMPose is in `xyxy` instead of `xywh`, which is consistent with the format used in other OpenMMLab projects like [MMDetection](https://github.com/open-mmlab/mmdetection).  We provide useful utils for bbox format conversion, such as `bbox_xyxy2xywh`, `bbox_xywh2xyxy`, `bbox_xyxy2cs`, etc., which are defined in `$MMPOSE/mmpose/structures/bbox/transforms.py`.
```

Let's take the implementation of the MPII dataset (`$MMPOSE/mmpose/datasets/datasets/body/mpii_dataset.py`) as an example.

```Python
@DATASETS.register_module()
class MpiiDataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/mpii.py')

    def __init__(self,
                 ## omitted
                 headbox_file: Optional[str] = None,
                 ## omitted
                ):

        if headbox_file:
            if data_mode != 'topdown':
                raise ValueError(
                    f'{self.__class__.__name__} is set to {data_mode}: '
                    'mode, while "headbox_file" is only '
                    'supported in topdown mode.')

            if not test_mode:
                raise ValueError(
                    f'{self.__class__.__name__} has `test_mode==False` '
                    'while "headbox_file" is only '
                    'supported when `test_mode==True`.')

            headbox_file_type = headbox_file[-3:]
            allow_headbox_file_type = ['mat']
            if headbox_file_type not in allow_headbox_file_type:
                raise KeyError(
                    f'The head boxes file type {headbox_file_type} is not '
                    f'supported. Should be `mat` but got {headbox_file_type}.')
        self.headbox_file = headbox_file

        super().__init__(
            ## omitted
            )

    def _load_annotations(self) -> List[dict]:
        """Load data from annotations in MPII format."""
        check_file_exist(self.ann_file)
        with open(self.ann_file) as anno_file:
            anns = json.load(anno_file)

        if self.headbox_file:
            check_file_exist(self.headbox_file)
            headbox_dict = loadmat(self.headbox_file)
            headboxes_src = np.transpose(headbox_dict['headboxes_src'],
                                         [2, 0, 1])
            SC_BIAS = 0.6

        data_list = []
        ann_id = 0

        # mpii bbox scales are normalized with factor 200.
        pixel_std = 200.

        for idx, ann in enumerate(anns):
            center = np.array(ann['center'], dtype=np.float32)
            scale = np.array([ann['scale'], ann['scale']],
                             dtype=np.float32) * pixel_std

            # Adjust center/scale slightly to avoid cropping limbs
            if center[0] != -1:
                center[1] = center[1] + 15. / pixel_std * scale[1]

            # MPII uses matlab format, index is 1-based,
            # we should first convert to 0-based index
            center = center - 1

            # unify shape with coco datasets
            center = center.reshape(1, -1)
            scale = scale.reshape(1, -1)
            bbox = bbox_cs2xyxy(center, scale)

            # load keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
            keypoints = np.array(ann['joints']).reshape(1, -1, 2)
            keypoints_visible = np.array(ann['joints_vis']).reshape(1, -1)

            data_info = {
                'id': ann_id,
                'img_id': int(ann['image'].split('.')[0]),
                'img_path': osp.join(self.data_prefix['img'], ann['image']),
                'bbox_center': center,
                'bbox_scale': scale,
                'bbox': bbox,
                'bbox_score': np.ones(1, dtype=np.float32),
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
            }

            if self.headbox_file:
                # calculate the diagonal length of head box as norm_factor
                headbox = headboxes_src[idx]
                head_size = np.linalg.norm(headbox[1] - headbox[0], axis=0)
                head_size *= SC_BIAS
                data_info['head_size'] = head_size.reshape(1, -1)

            data_list.append(data_info)
            ann_id = ann_id + 1

        return data_list
```

When supporting MPII dataset, since we need to use `head_size` to calculate `PCKh`, we add `headbox_file` to `__init__()` and override`_load_annotations()`.

To support a dataset that is beyond the scope of `BaseCocoStyleDataset`, you may need to subclass from the `BaseDataset` provided by [MMEngine](https://github.com/open-mmlab/mmengine). Please refer to the [documents](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html) for details.

### Pipeline

Data augmentations and transformations during pre-processing are organized as a pipeline. Here is an example of typical pipelines：

```Python
# pipelines
train_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', target_type='heatmap', encoder=codec),
    dict(type='PackPoseInputs')
]
test_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
```

In a keypoint detection task, data will be transformed among three scale spaces:

- **Original Image Space**: the space where the images are stored. The sizes of different images are not necessarily the same

- **Input Image Space**: the image space used for model input. All **images** and **annotations** will be transformed into this space, such as `256x256`, `256x192`, etc.

- **Output Space**: the scale space where model outputs are located, such as `64x64(Heatmap)`，`1x1(Regression)`, etc. The supervision signal is also in this space during training

Here is a diagram to show the workflow of data transformation among the three scale spaces:

![migration-en](https://user-images.githubusercontent.com/13503330/187190213-cad87b5f-0a95-4f1f-b722-15896914ded4.png)

In MMPose, the modules used for data transformation are under `$MMPOSE/mmpose/datasets/transforms`, and their workflow is shown as follows:

![transforms-en](https://user-images.githubusercontent.com/13503330/187190352-a7662346-b8da-4256-9192-c7a84b15cbb5.png)

#### i. Augmentation

Commonly used transforms are defined in `$MMPOSE/mmpose/datasets/transforms/common_transforms.py`, such as `RandomFlip`, `RandomHalfBody`, etc.

For top-down methods, `Shift`, `Rotate`and `Resize` are implemented by `RandomBBoxTransform`**.** For bottom-up methods, `BottomupRandomAffine` is used.

```{note}
Most data transforms depend on `bbox_center` and `bbox_scale`, which can be obtained by `GetBBoxCenterScale`.
```

#### ii. Transformation

Affine transformation is used to convert images and annotations from the original image space to the input space. This is done by `TopdownAffine` for top-down methods and `BottomupRandomAffine` for bottom-up methods.

#### iii. Encoding

In training phase, after the data is transformed from the original image space into the input space, it is necessary to use `GenerateTarget` to obtain the training target(e.g. Gaussian Heatmaps). We name this process **Encoding**. Conversely, the process of getting the corresponding coordinates from Gaussian Heatmaps is called **Decoding**.

In MMPose, we collect Encoding and Decoding processes into a **Codec**, in which `encode()` and `decode()` are implemented.

Currently we support the following types of Targets.

- `heatmap`: Gaussian heatmaps
- `keypoint_label`: keypoint representation (e.g. normalized coordinates)
- `keypoint_xy_label`: axis-wise keypoint representation
- `heatmap+keypoint_label`: Gaussian heatmaps and keypoint representation
- `multiscale_heatmap`: multi-scale Gaussian heatmaps

and the generated targets will be packed as follows.

- `heatmaps`: Gaussian heatmaps
- `keypoint_labels`: keypoint representation (e.g. normalized coordinates)
- `keypoint_x_labels`: keypoint x-axis representation
- `keypoint_y_labels`: keypoint y-axis representation
- `keypoint_weights`: keypoint visibility and weights

Note that we unify the data format of top-down and bottom-up methods, which means that a new dimension is added to represent different instances from the same image, in shape:

```Python
[batch_size, num_instances, num_keypoints, dim_coordinates]
```

- top-down: `[B, 1, K, D]`

- Bottom-up: `[B, N, K, D]`

The provided codecs are stored under `$MMPOSE/mmpose/codecs`.

```{note}
If you wish to customize a new codec, you can refer to [Codec](./user_guides/codecs.md) for more details.
```

#### iv. Packing

After the data is transformed, you need to pack it using `PackPoseInputs`.

This method converts the data stored in the dictionary `results` into standard data structures in MMPose, such as `InstanceData`, `PixelData`, `PoseDataSample`, etc.

Specifically, we divide the data into `gt` (ground-truth) and `pred` (prediction), each of which has the following types:

- **instances**(numpy.array): instance-level raw annotations or predictions in the original scale space
- **instance_labels**(torch.tensor): instance-level training labels (e.g. normalized coordinates, keypoint visibility) in the output scale space
- **fields**(torch.tensor): pixel-level training labels or predictions (e.g. Gaussian Heatmaps) in the output scale space

The following is an example of the implementation of `PoseDataSample` under the hood:

```Python
def get_pose_data_sample(self):
    # meta
    pose_meta = dict(
        img_shape=(600, 900),   # [h, w, c]
        crop_size=(256, 192),   # [h, w]
        heatmap_size=(64, 48),  # [h, w]
    )

    # gt_instances
    gt_instances = InstanceData()
    gt_instances.bboxes = np.random.rand(1, 4)
    gt_instances.keypoints = np.random.rand(1, 17, 2)

    # gt_instance_labels
    gt_instance_labels = InstanceData()
    gt_instance_labels.keypoint_labels = torch.rand(1, 17, 2)
    gt_instance_labels.keypoint_weights = torch.rand(1, 17)

    # pred_instances
    pred_instances = InstanceData()
    pred_instances.keypoints = np.random.rand(1, 17, 2)
    pred_instances.keypoint_scores = np.random.rand(1, 17)

    # gt_fields
    gt_fields = PixelData()
    gt_fields.heatmaps = torch.rand(17, 64, 48)

    # pred_fields
    pred_fields = PixelData()
    pred_fields.heatmaps = torch.rand(17, 64, 48)
    data_sample = PoseDataSample(
        gt_instances=gt_instances,
        pred_instances=pred_instances,
        gt_fields=gt_fields,
        pred_fields=pred_fields,
        metainfo=pose_meta)

    return data_sample
```

## Step3: Model

In MMPose 1.0, the model consists of the following components:

- **Data Preprocessor**: perform data normalization and channel transposition

- **Backbone**: used for feature extraction

- **Neck**: GAP，FPN, etc. are optional

- **Head**: used to implement the core algorithm and loss function

We define a base class `BasePoseEstimator` for the model in `$MMPOSE/models/pose_estimators/base.py`. All models, e.g. `TopdownPoseEstimator`, should inherit from this base class and override the corresponding methods.

Three modes are provided in `forward()` of the estimator:

- `mode == 'loss'`: return the result of loss function for model training

- `mode == 'predict'`: return the prediction result in the input space, used for model inference

- `mode == 'tensor'`: return the model output in the output space, i.e. model forward propagatin only, for model export

Developers should build the components by calling the corresponding registry. Taking the top-down model as an example:

```Python
@MODELS.register_module()
class TopdownPoseEstimator(BasePoseEstimator):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)
```

### Data Preprocessor

Starting from MMPose 1.0, we have added a new module to the model called data preprocessor, which performs data preprocessings like image normalization and channel transposition. It can benefit from the high computing power of devices like GPU, and improve the integrity in model export and deployment.

A typical `data_preprocessor` in the config is as follows:

```Python
data_preprocessor=dict(
    type='PoseDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True),
```

It will transpose the channel order of the input image from `bgr` to `rgb` and normalize the data according to `mean` and `std`.

### Backbone

MMPose provides some commonly used backbones under `$MMPOSE/mmpose/models/backbones`.

In practice, developers often use pre-trained backbone weights for transfer learning, which can improve the performance of the model on small datasets.

In MMPose, you can use the pre-trained weights by setting `init_cfg` in config:

```Python
init_cfg=dict(
    type='Pretrained',
    checkpoint='PATH/TO/YOUR_MODEL_WEIGHTS.pth'),
```

If you want to load a checkopoint to your backbone, you should specify the `prefix`:

```Python
init_cfg=dict(
    type='Pretrained',
    prefix='backbone.',
    checkpoint='PATH/TO/YOUR_CHECKPOINT.pth'),
```

`checkpoint` can be either a local path or a download link. Thus, if you wish to use a pre-trained model provided by Torchvision(e.g. ResNet50), you can simply use:

```Python
init_cfg=dict(
    type='Pretrained',
    checkpoint='torchvision://resnet50')
```

In addition to these commonly used backbones, you can easily use backbones from other repositories in the OpenMMLab family such as MMClassification, which all share the same config system and provide pre-trained weights.

It should be emphasized that if you add a new backbone, you need to register it by doing:

```Python
@MODELS.register_module()
class YourBackbone(BaseBackbone):
```

Besides, import it in `$MMPOSE/mmpose/models/backbones/__init__.py`, and add it to `__all__`.

### Neck

Neck is usually a module between Backbone and Head, which is used in some algorithms. Here are some commonly used Neck:

- Global Average Pooling(GAP)

- Feature Pyramid Networks(FPN)

### Head

Generally speaking, Head is often the core of an algorithm, which is used to make predictions and perform loss calculation.

Modules related to Head in MMPose are defined under `$MMPOSE/mmpose/models/heads`, and developers need to inherit the base class `BaseHead` when customizing Head and override the following methods:

- forward()

- predict()

- loss()

Specifically, `predict()` method needs to return pose predictions in the image space, which is obtained from the model output though the decoding function provided by the codec. We implement this process in `BaseHead.decode()`.

On the other hand, we will perform test-time augmentation(TTA) in `predict()`.

A commonly used TTA is `flip_test`, namely, an image and its flipped version are sent into the model to inference, and the output of the flipped version will be flipped back, then average them to stabilize the prediction.

Here is an example of `predict()` in `RegressionHead`:

```Python
def predict(self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            test_cfg: ConfigType = {}) -> Predictions:
    """Predict results from outputs."""

    if test_cfg.get('flip_test', False):
        # TTA: flip test -> feats = [orig, flipped]
        assert isinstance(feats, list) and len(feats) == 2
        flip_indices = batch_data_samples[0].metainfo['flip_indices']
        input_size = batch_data_samples[0].metainfo['input_size']
        _feats, _feats_flip = feats
        _batch_coords = self.forward(_feats)
        _batch_coords_flip = flip_coordinates(
            self.forward(_feats_flip),
            flip_indices=flip_indices,
            shift_coords=test_cfg.get('shift_coords', True),
            input_size=input_size)
        batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
    else:
        batch_coords = self.forward(feats)  # (B, K, D)

    batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
    preds = self.decode(batch_coords)
```

The `loss()` not only performs the calculation of loss functions, but also the calculation of training-time metrics such as pose accuracy. The results are carried by a dictionary `losses`:

```Python
 # calculate accuracy
_, avg_acc, _ = keypoint_pck_accuracy(
    pred=to_numpy(pred_coords),
    gt=to_numpy(keypoint_labels),
    mask=to_numpy(keypoint_weights) > 0,
    thr=0.05,
    norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
losses.update(acc_pose=acc_pose)
```

The data of each batch is packaged into `batch_data_samples`. Taking the Regression-based method as an example, the normalized coordinates and keypoint weights can be obtained as follows:

```Python
keypoint_labels = torch.cat(
    [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
keypoint_weights = torch.cat([
    d.gt_instance_labels.keypoint_weights for d in batch_data_samples
])
```

Here is the complete implementation of `loss()` in `RegressionHead`:

```Python
def loss(self,
         inputs: Tuple[Tensor],
         batch_data_samples: OptSampleList,
         train_cfg: ConfigType = {}) -> dict:
    """Calculate losses from a batch of inputs and data samples."""

    pred_outputs = self.forward(inputs)

    keypoint_labels = torch.cat(
        [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
    keypoint_weights = torch.cat([
        d.gt_instance_labels.keypoint_weights for d in batch_data_samples
    ])

    # calculate losses
    losses = dict()
    loss = self.loss_module(pred_outputs, keypoint_labels,
                            keypoint_weights.unsqueeze(-1))

    if isinstance(loss, dict):
        losses.update(loss)
    else:
        losses.update(loss_kpt=loss)

    # calculate accuracy
    _, avg_acc, _ = keypoint_pck_accuracy(
        pred=to_numpy(pred_outputs),
        gt=to_numpy(keypoint_labels),
        mask=to_numpy(keypoint_weights) > 0,
        thr=0.05,
        norm_factor=np.ones((pred_outputs.size(0), 2), dtype=np.float32))
    acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
    losses.update(acc_pose=acc_pose)

    return losses
```

## Compatibility of MMPose 0.X

MMPose 1.0 has been refactored extensively and addressed many legacy issues. Most of the code in MMPose 1.0 will not be compatible with 0.x version.

To try our best to help you migrate your code and model, here are some major changes:

### Data Transformation

#### Translation, Rotation and Scaling

The transformation methods `TopDownRandomShiftBboxCenter` and `TopDownGetRandomScaleRotation` in old version, will be merged into `RandomBBoxTransform`.

```Python
@TRANSFORMS.register_module()
class RandomBBoxTransform(BaseTransform):
    r"""Rnadomly shift, resize and rotate the bounding boxes.

    Required Keys:

        - bbox_center
        - bbox_scale

    Modified Keys:

        - bbox_center
        - bbox_scale

    Added Keys:
        - bbox_rotation

    Args:
        shift_factor (float): Randomly shift the bbox in range
            :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
            where :math:`dx(y) = x(y)_scale \cdot shift_factor` in pixels.
            Defaults to 0.16
        shift_prob (float): Probability of applying random shift. Defaults to
            0.3
        scale_factor (Tuple[float, float]): Randomly resize the bbox in range
            :math:`[scale_factor[0], scale_factor[1]]`. Defaults to (0.5, 1.5)
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-rotate_factor, rotate_factor]` in degrees. Defaults
            to 80.0
        rotate_prob (float): Probability of applying random rotation. Defaults
            to 0.6
    """

    def __init__(self,
                 shift_factor: float = 0.16,
                 shift_prob: float = 0.3,
                 scale_factor: Tuple[float, float] = (0.5, 1.5),
                 scale_prob: float = 1.0,
                 rotate_factor: float = 80.0,
                 rotate_prob: float = 0.6) -> None:
```

#### Target Generation

The old methods like:

- `TopDownGenerateTarget`
- `TopDownGenerateTargetRegression`
- `BottomUpGenerateHeatmapTarget`
- `BottomUpGenerateTarget`

will be merged in to `GenerateTarget`, and the actual generation methods are implemented in [Codec](./user_guides/codecs.md).

```Python
@TRANSFORMS.register_module()
class GenerateTarget(BaseTransform):
    """Encode keypoints into Target.

    The generated target is usually the supervision signal of the model
    learning, e.g. heatmaps or regression labels.

    Required Keys:

        - keypoints
        - keypoints_visible
        - dataset_keypoint_weights

    Added Keys (depends on the args):
        - heatmaps
        - keypoint_labels
        - keypoint_x_labels
        - keypoint_y_labels
        - keypoint_weights

    Args:
        encoder (dict | list[dict]): The codec config for keypoint encoding
        target_type (str): The type of the encoded form of the keypoints.
            Should be one of the following options:

            - ``'heatmap'``: The encoded should be instance-irrelevant
                heatmaps and will be stored in ``results['heatmaps']``
            - ``'multiscale_heatmap'`` The encoded should be a list of
                heatmaps and will be stored in ``results['heatmaps']``. Note
                that in this case ``self.encoder`` is also a list, each
                encoder for a single scale of heatmaps
            - ``'keypoint_label'``: The encoded should be instance-level
                labels and will be stored in ``results['keypoint_label']``
            - ``'keypoint_xy_label'``: The encoed should be instance-level
                labels in x-axis and y-axis respectively. They will be stored
                in ``results['keypoint_x_label']`` and
                ``results['keypoint_y_label']``
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
    """

    def __init__(self,
                 encoder: MultiConfig,
                 target_type: str,
                 use_dataset_keypoint_weights: bool = False) -> None:
```

#### Data Normalization

The data normalization operations `NormalizeTensor` and `ToTensor` will be replaced by **DataPreprocessor** module, which will no longer be used as a preprocessing operation, but will be merged as a part of the model forward propagation.

### Compatibility of Models

We have performed compatibility with the model weights provided by model zoo to ensure that the same model weights can get a comparable accuracy in both version. But note that due to the large number of differences in processing details, the inference outputs can be slightly different(less than 0.05% difference in accuracy).

For model weights saved by training with 0.x version, we provide a `_load_state_dict_pre_hook()` method in Head to replace the old version of the `state_dict` with the new one. If you wish to make your model compatible with MMPose 1.0, you can refer to our implementation as follows.

```Python
@MODELS.register_module()
class YourHead(BaseHead):
def __init__(self):

    ## omitted

    # Register the hook to automatically convert old version state dicts
    self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
```

#### Heatmap-based Model

For models based on `SimpleBaseline` approach, developers need to pay attention to the last convolutional layer.

```Python
def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                              **kwargs):
    version = local_meta.get('version', None)

    if version and version >= self._version:
        return

    # convert old-version state dict
    keys = list(state_dict.keys())
    for _k in keys:
        if not _k.startswith(prefix):
            continue
        v = state_dict.pop(_k)
        k = _k[len(prefix):]
        # In old version, "final_layer" includes both intermediate
        # conv layers (new "conv_layers") and final conv layers (new
        # "final_layer").
        #
        # If there is no intermediate conv layer, old "final_layer" will
        # have keys like "final_layer.xxx", which should be still
        # named "final_layer.xxx";
        #
        # If there are intermediate conv layers, old "final_layer"  will
        # have keys like "final_layer.n.xxx", where the weights of the last
        # one should be renamed "final_layer.xxx", and others should be
        # renamed "conv_layers.n.xxx"
        k_parts = k.split('.')
        if k_parts[0] == 'final_layer':
            if len(k_parts) == 3:
                assert isinstance(self.conv_layers, nn.Sequential)
                idx = int(k_parts[1])
                if idx < len(self.conv_layers):
                    # final_layer.n.xxx -> conv_layers.n.xxx
                    k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                else:
                    # final_layer.n.xxx -> final_layer.xxx
                    k_new = 'final_layer.' + k_parts[2]
            else:
                # final_layer.xxx remains final_layer.xxx
                k_new = k
        else:
            k_new = k

        state_dict[prefix + k_new] = v
```

#### RLE-based Model

For the RLE-based models, since the loss module is renamed to `loss_module` in MMPose 1.0, and the flow model is subsumed under the loss module, changes need to be made to the keys in `state_dict`:

```Python
def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                              **kwargs):

    version = local_meta.get('version', None)

    if version and version >= self._version:
        return

    # convert old-version state dict
    keys = list(state_dict.keys())
    for _k in keys:
        v = state_dict.pop(_k)
        k = _k.lstrip(prefix)
        # In old version, "loss" includes the instances of loss,
        # now it should be renamed "loss_module"
        k_parts = k.split('.')
        if k_parts[0] == 'loss':
            # loss.xxx -> loss_module.xxx
            k_new = prefix + 'loss_module.' + '.'.join(k_parts[1:])
        else:
            k_new = _k

        state_dict[k_new] = v
```
