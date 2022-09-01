# 迁移指南

重构之后的 MMPose1.0 与之前的版本有较大改动，对部分模块进行了重新设计和组织，降低代码冗余度，提升运行效率，降低学习难度。

对于有一定基础的开发者，本章节提供了一份迁移指南。不论你是**旧版MMPose的用户**，还是**希望将自己的Pytorch项目迁移到MMPose的新用户**，都可以通过本教程了解如何构建一个基于 MMPose1.0 的项目。

```{note}
本教程包含了使用 MMPose1.0 时开发者会关心的内容：

- 整体代码架构与设计逻辑

- 如何用config文件管理模块

- 如何使用自定义数据集

- 如何添加新的模块（骨干网络、模型头部、损失函数等）
```

## 整体架构与设计

![overall-cn](https://user-images.githubusercontent.com/13503330/187830967-f2d7bf40-6261-42f3-91a5-ae045fa0dc0c.png)

一般来说，开发者在项目开发过程中经常接触内容的主要有**五个**方面：

- **通用**：环境、Hook、Checkpoint、Logger、Timer等

- **数据**：Dataset、Dataloader、数据增强等

- **训练**：优化器、学习率调整等

- **模型**：Backbone、Neck、Head、损失函数等

- **评测**：Metrics

其中**通用**、**训练**和**评测**相关的模块往往由训练框架提供，开发者只需要调用和调整参数，不需要自行实现，开发者主要实现的是**数据**和**模型**部分。

## Step1：配置文件

在MMPose中，我们使用一个python文件作为config，用于整个项目的定义、参数管理，因此我们强烈建议第一次接触MMPose的开发者，查阅 [配置文件](./user_guides/configs.md) 学习配置文件的定义。

需要注意的是，所有新增的模块都需要使用`Registry`进行注册，并在对应目录的`__init__.py`中进行`import`。

## Step2：数据

MMPose数据的组织主要包含三个方面：

- 数据格式

- 数据集

- 数据流水线

### 数据格式

在MMPose中，**所有数据都使用COCO风格进行组织**，我们在`$MMPOSE/mmpose/datasets/base`下定义了一个基类`BaseCocoStyleDataset`。

bbox的数据格式采用`xyxy`，而不是`xywh`，这与`mmdet`中采用的格式一致。

如果你的数据原本就是使用COCO格式进行存储的，那么可以直接使用我们的实现。

当你的数据不是用COCO格式存储时，你需要在`$MMPOSE/configs/_base_/datasets`目录下定义数据的关键点信息（关键点顺序、骨架信息、权重、标注信息的标准差）。

对于不同bbox格式之间的转换，我们同样提供了丰富的方法：`bbox_xyxy2xywh`、`bbox_xywh2xyxy`、`bbox_xyxy2cs`等，定义在`$MMPOSE/mmpose/structures/bbox/transforms.py`中，可以帮助你完成自己数据格式的转换。

```{note}
关于COCO数据格式的详细说明请参考[COCO]。
```

以MPII数据集（`$MMPOSE/configs/_base_/datasets/mpii.py`）为例：

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
        ## 内容省略
    },
    skeleton_info={
        0:
        dict(link=('right_ankle', 'right_knee'), id=0, color=[255, 128, 0]),
        ## 内容省略
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

### 数据集

当你的数据不是用COCO格式存储时，你需要在`$MMPOSE/mmpose/datasets/datasets`目录下实现Dataset的定义，将数据组织为COCO格式。

下面我们以MPII数据集的实现（`$MMPOSE/mmpose/datasets/datasets/body/mpii_dataset.py`）为例：

```Python
@DATASETS.register_module()
class MpiiDataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/mpii.py')

    def __init__(self,
                 ## 内容省略
                 headbox_file: Optional[str] = None,
                 ## 内容省略):

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
            ## 内容省略
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

在对MPII数据集进行支持时，由于MPII需要读入`head_size`信息来计算`PCKh`，因此我们在`__init__()`中增加了`headbox_file`，并重载了`_load_annotations()`来完成数据组织。

### 数据流水线

一个典型的数据流水线配置如下：

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

在关键点检测任务中，数据一般会在三个尺度空间中变换：

- **原始图片空间**：图片存储时的原始空间，不同图片的尺寸不一定相同

- **输入图片空间**：用于模型训练的图片尺度空间，所有**图片**和**标注**被缩放到输入尺度，如`256x256`，`256x192`等

- **输出尺度空间**：用于模型训练的标注尺度空间，同时也是模型预测结果所在的尺度空间，如`64x64(Heatmap)`，`1x1(Regression)`等

数据在三个空间中变换的流程如图所示：

![migration-cn](https://user-images.githubusercontent.com/13503330/187831574-13804daf-f498-47c2-ba43-64b8e6ffe3dd.png)

在MMPose中，数据变换所需要的模块在`$MMPOSE/mmpose/datasets/transforms`目录下，它们的工作流程如图所示：

![transforms-cn](https://user-images.githubusercontent.com/13503330/187831611-8db89e20-95c7-42bc-8b0d-700fadf60328.png)

#### i. 数据增强

数据增强中常用的变换存放在`$MMPOSE/mmpose/datasets/transforms/common_transforms.py`中，如`RandomFlip`、`RandomHalfBody`等。

对于top-down方法，`Shift`、`Rotate`、`Resize`操作由`RandomBBoxTransform`来实现，对于bottom-up方法，则是由`BottomupRandomAffine`实现。

值得注意的是，大部分数据变换都依赖于`bbox_center`和`bbox_scale`，可以通过`GetBBoxCenterScale`来得到。

```{note}
这部分所有操作只会生成对应的**变换矩阵**，**不会**对原始数据进行实际的变换。
```

#### ii. 数据变换

当完成对应的变换矩阵生成后，我们会通过仿射变换来实际变换图片和标注。

对于top-down方法，是由`TopdownAffine`来完成。对于bottom-up方法，是在`BottomupRandomAffine`中完成。

#### iii. 数据编码

数据从原始空间变换到输入空间后，需要使用`GenerateTarget`来生成训练所需的Target（比如用坐标值生成高斯热图），我们将这一过程称为编码（Encode），反之，通过高斯热图得到对应坐标值的过程称为解码（Decode）。

在MMPose中，我们将编码和解码过程集合成一个编解码器（Codec），在其中实现`encode()`和`decode()`。

目前MMPose支持以下类型的Target：

- `heatmaps`：高斯热图

- `keypoint_labels`：归一化的坐标值

- `keypoint_x_labels`：x轴坐标表征

- `keypoint_y_labels`：y轴坐标表征

- `keypoint_weights`：关键点权重

```Python
@TRANSFORMS.register_module()
class GenerateTarget(BaseTransform):
    """Encode keypoints into Target.

    Added Keys (depends on the args):
        - heatmaps
        - keypoint_labels
        - keypoint_x_labels
        - keypoint_y_labels
        - keypoint_weights
    """
```

值得注意的是，我们对top-down和bottom-up的数据格式进行了统一，这意味着标注信息中会新增一个维度来代表同一张图里的不同instance，格式为：

```Python
[batch_size, num_instances, num_keypoints, dim_coordinates]
```

- top-down：`[B, 1, K, D]`

- Bottom-up: `[B, N, K, D]`

当前已经支持的编解码器定义在`$MMPOSE/mmpose/codecs`目录下，如果你需要自定新的编解码器，可以前往 [编解码器](./user_guides/codecs.md) 了解更多详情。

#### iv. 数据打包

数据经过变换完成后，都需要通过`PackPoseInputs`进行打包，转换成MMPose训练所需要的格式，定义在`$MMPOSE/mmpose/datasets/transforms/formatting.py`中。

这一方法会将数据流水线中用字典`results`存储的数据转换成用MMEngine训练所需的`InstanceData`，`PixelData`，`PoseDataSample`格式。

如果你的模型训练所需的格式超出了MMPose支持的范围，那么你需要注意变换过程中`results`、`PackPoseInputs`和`PoseDataSample`。

具体而言，我们将数据分为`gt`和`pred`两种，每一种都有如下类型：

- **instances**(numpy.array(：实例级别的原始标注，用于在原始尺度空间下进行模型评测

- **instance_labels**(torch.tensor)：实例级别的训练标签（如归一化的坐标值、关键点可见性），用于在输出尺度空间下进行模型训练

- **fields**(torch.tensor)：实例级别且带空间信息的训练标签（如高斯热图），用于在输出尺度空间下进行模型训练

下面是`PoseDataSample`底层实现的例子：

```Python
def get_pose_data_sample(self):
    # meta
    pose_meta = dict(
        img_shape=(600, 900),  # [h, w, c]
        crop_size=(256, 192),  # [h, w]
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

## Step3: 模型

在MMPose1.0中，我们的模型由以下几部分构成：

- **Data Preprocessor**：完成数据归一化和通道转换

- **Backbone**：骨干网络，用于特征提取

- **Neck**：GAP，FPN等可选项

- **Head**：模型头部，用于实现核心算法功能和损失函数定义

我们在`$MMPOSE/models/pose_estimators/base.py`下为姿态估计模型定义了一个基类`BasePoseEstimator`，所有的模型都需要继承这个基类，并重载对应的方法。

根据算法流程，MMPose将模型划分为`TopdownPoseEstimator`、`BottomupPoseEstimator`等，在`forward()`中提供了三种不同的模式：

- `mode == 'loss'`：返回损失函数计算的结果，用于模型训练

- `mode == 'predict'`：返回输入尺度下的预测结果，用于模型推理

- `mode == 'tensor'`：返回输出尺度下的模型输出，即只进行模型前向传播，用于模型导出

开发者需要在`PoseEstimator`中按照模型结构调用对应的`Registry`，对模块进行实例化。以top-down模型为例：

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

从 MMPose1.0 开始，我们将数据归一化和通道转换操作作为模块加入到模型结构中，这样做的好处是可以进一步实现模型端到端训练和预测，让训练好的模型可以直接以图片作为输入，而不需要用户自己实现数据归一化预处理。

在配置文件中，一个常见的`data_preprocessor`如下：

```Python
data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
```

它会将输入图片的通道顺序从`bgr`转换为`rgb`，并根据`mean`和`std`进行数据归一化。

### Backbone

MMPose实现的backbone存放在`$MMPOSE/mmpose/models/backbones`目录下。

在实际开发中，开发者经常会使用预训练的backbone权重进行迁移学习，这能有效提升模型在小数据集上的性能。

在MMPose中，我们只需要在配置文件backbone的`init_cfg`中设置：

```Python
init_cfg=dict(
    type='Pretrained',
    checkpoint='PATH/TO/YOUR_MODEL_WEIGHTS.pth'),
```

其中`checkpoint`既可以是本地路径，也可以是下载链接。因此，如果你想使用Torchvision提供的预训练模型（比如ResNet50），可以使用：

```Python
init_cfg=dict(
    type='Pretrained',
    checkpoint='torchvision://resnet50')
```

除了这些常用的backbone以外，你还可以从MMClassification等OpenMMLab生态系统中的仓库，方便地迁移backbone，它们都遵循同一套配置文件格式，并提供了预训练权重可供使用。

需要强调的是，如果你加入了新的backbone，需要在模型定义时进行注册：

```Python
@MODELS.register_module()
class YourBackbone(BaseBackbone):
```

同时在`$MMPOSE/mmpose/models/backbones/__init__.py`下进行`import`，并加入到`__all__`中，才能被配置文件正确地调用。

### Neck

Neck通常是介于backbone和head之间的模块，在有些模型算法中会用到，常见的Neck有：

- Global Average Pooling(GAP)

- Feature Pyramid Networks(FPN)

### Head

通常来说，Head是模型算法实现的核心，用于控制模型的输出，并进行Loss计算。

MMPose中Head相关的模块定义在`$MMPOSE/mmpose/models/heads`目录下，开发者在自定义Head时需要继承我们提供的基类`BaseHead`，并重载以下三个方法对应模型推理的三种模式：

- forward()

- predict()

- loss()

具体而言，`predict()`返回的应是输入图片尺度下的结果，因此需要调用`self.decode()`对输出进行解码，我们在`BaseHead`中已经进行了实现，它会调用编解码器提供的`decoder`来完成解码过程。

另一方面，我们会在`predict()`中进行测试时增强。在进行预测时，一个常见的测试时增强技巧是进行翻转集成。即，将一张图片先进行一次推理，再将图片水平翻转进行一次推理，推理的结果再次水平翻转回去，对两次推理的结果进行平均。这个技巧能有效提升模型的预测稳定性。

下面是在`RegressionHead`中定义`predict()`的例子：

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

`loss()`除了进行损失函数的计算，还会进行accuracy等训练时指标的计算，并通过一个字典`losses`来传递:

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

每个batch的数据都打包成了`batch_data_samples`，你可以根据targert类型获取，以Regression-based方法为例，训练所需的归一化的坐标值和关键点权重可以用如下方式获取：

```Python
keypoint_labels = torch.cat(
    [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
keypoint_weights = torch.cat([
    d.gt_instance_labels.keypoint_weights for d in batch_data_samples
])
```

以下为`RegressionHead`中完整的`loss()`实现：

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

## MMPose 0.X 兼容性说明

MMPose 1.0 经过了大规模重构并解决了许多遗留问题，对于0.x版本的大部分代码 MMPose 1.0 将不兼容。

### 数据变换

#### 平移、旋转和缩放

旧版的数据变换方法`TopDownRandomShiftBboxCenter`和`TopDownGetRandomScaleRotation`，将被合并为`RandomBBoxTransform`：

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

#### 标签生成

旧版用于训练标签生成的方法`TopDownGenerateTarget`、`TopDownGenerateTargetRegression`、`BottomUpGenerateHeatmapTarget`、`BottomUpGenerateTarget`等将被合并为`GenerateTarget`，而实际的生成方法由 [编解码器](./user_guides/codecs.md)提供：

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

#### 数据归一化

旧版的数据归一化操作`NormalizeTensor`和`ToTensor`方法将由**DataPreprocessor**模块替代，不再作为预处理操作，而是作为模块加入到模型前向传播中。

### 模型兼容

我们对model zoo提供的模型权重进行了兼容性处理，确保相同的模型权重测试精度能够与0.x版本保持同等水平，但由于在这两个版本中存在大量处理细节的差异，推理可能会产生轻微的不同（误差小于0.05%）。

对于使用0.x版本训练保存的模型权重，我们在Head中提供了一个`_load_state_dict_pre_hook()`方法来将旧版的权重字典替换为新版，如果你希望将在旧版上开发的模型兼容到新版，可以参考我们的实现。

```Python
@MODELS.register_module()
class YourHead(BaseHead):
def __init__(self):

    ## omitted

    # Register the hook to automatically convert old version state dicts
    self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
```

#### Heatmap-based方法

对于基于SimpleBaseline方法的模型，主要需要注意最后一层卷积层的兼容：

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

#### RLE-based方法

对于基于RLE的模型，由于新版的`loss`模块更名为`loss_module`，且flow模型归属在`loss`模块下，因此需要对权重字典中`loss`字段进行更改：

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
