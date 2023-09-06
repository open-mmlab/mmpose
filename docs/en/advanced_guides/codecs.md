# Learn about Codecs

In the keypoint detection task, depending on the algorithm, it is often necessary to generate targets in different formats, such as normalized coordinates, vectors and heatmaps, etc. Similarly, for the model outputs, a decoding process is required to transform them into coordinates.

Encoding and decoding are closely related and inverse each other. In earlier versions of MMPose, encoding and decoding are implemented at different modules, making it less intuitive and unified.

MMPose 1.0 introduced a new module **Codec** to integrate the encoding and decoding together in a modular and user-friendly form.

Here is a diagram to show where the `Codec` is:

![pose_estimator_en](https://github.com/open-mmlab/mmpose/assets/13503330/0764baab-41c7-4a1d-ab64-5d7f9dfc8eec)

## Basic Concepts

A typical codec consists of two parts:

- Encoder
- Decoder

### Encoder

The encoder transforms the coordinates in the input image space into the needed target format:

- Normalized Coordinates
- One-dimensional Vectors
- Gaussian Heatmaps

For example, in the Regression-based method, the encoder will be:

```Python
def encode(self,
           keypoints: np.ndarray,
           keypoints_visible: Optional[np.ndarray] = None) -> dict:
    """Encoding keypoints from input image space to normalized space.

    Args:
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)

    Returns:
        dict:
        - keypoint_labels (np.ndarray): The normalized regression labels in
            shape (N, K, D) where D is 2 for 2d coordinates
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)
    """
    if keypoints_visible is None:
        keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

    w, h = self.input_size
    valid = ((keypoints >= 0) &
             (keypoints <= [w - 1, h - 1])).all(axis=-1) & (
                 keypoints_visible > 0.5)

    keypoint_labels = (keypoints / np.array([w, h])).astype(np.float32)
    keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

    encoded = dict(
        keypoint_labels=keypoint_labels, keypoint_weights=keypoint_weights)

    return encoded
```

The encoded data is converted to Tensor format in `PackPoseInputs` and packed in `data_sample.gt_instance_labels` for model calls. By default it will consist of the following encoded fields:

- `keypoint_labels`
- `keypoint_weights`
- `keypoints_visible_weights`

To specify data fields to be packed, you can define the `label_mapping_table` attribute in the codec. For example, in `VideoPoseLifting`:

```Python
label_mapping_table = dict(
        trajectory_weights='trajectory_weights',
        lifting_target_label='lifting_target_label',
        lifting_target_weight='lifting_target_weight',
)
```

`data_sample.gt_instance_labels` are generally used for loss calculation, as demonstrated by `loss()` in `RegressionHead`.

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

    losses.update(loss_kpt=loss)
    ### Omitted ###
```

```{note}
Encoder also defines data to be packed in `data_sample.gt_instances` and `data_sample.gt_fields`. Modify `instance_mapping_table` and `field_mapping_table` in the codec will specify values to be packed respectively. For default values, please check [BaseKeypointCodec](https://github.com/open-mmlab/mmpose/blob/main/mmpose/codecs/base.py).
```

### Decoder

The decoder transforms the model outputs into coordinates in the input image space, which is the opposite processing of the encoder.

For example, in the Regression-based method, the decoder will be:

```Python
def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode keypoint coordinates from normalized space to input image
    space.

    Args:
        encoded (np.ndarray): Coordinates in shape (N, K, D)

    Returns:
        tuple:
        - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
        - scores (np.ndarray): The keypoint scores in shape (N, K).
            It usually represents the confidence of the keypoint prediction

    """

    if encoded.shape[-1] == 2:
        N, K, _ = encoded.shape
        normalized_coords = encoded.copy()
        scores = np.ones((N, K), dtype=np.float32)
    elif encoded.shape[-1] == 4:
        # split coords and sigma if outputs contain output_sigma
        normalized_coords = encoded[..., :2].copy()
        output_sigma = encoded[..., 2:4].copy()
        scores = (1 - output_sigma).mean(axis=-1)
    else:
        raise ValueError(
            'Keypoint dimension should be 2 or 4 (with sigma), '
            f'but got {encoded.shape[-1]}')

    w, h = self.input_size
    keypoints = normalized_coords * np.array([w, h])

    return keypoints, scores
```

By default, the `decode()` method only performs decoding on a single instance. You can also implement the `batch_decode()` method to boost the decoding process.

## Common Usage

The example below shows how to use a codec in your config:

- Define the Codec
- Generate Targets
- Head

### Define the Codec

Take the Regression-based method to generate normalized coordinates as an example, you can define a `codec` in your config as follows:

```Python
codec = dict(type='RegressionLabel', input_size=(192, 256))
```

### Generate Targets

In pipelines, A codec should be passed into `GenerateTarget` to work as the `encoder`:

```Python
dict(type='GenerateTarget', encoder=codec)
```

### Head

In MMPose workflows, we decode the model outputs in `Head`, which requires a codec to work as the `decoder`:

```Python
head=dict(
    type='RLEHead',
    in_channels=2048,
    num_joints=17,
    loss=dict(type='RLELoss', use_target_weight=True),
    decoder=codec
)
```

Here is the phase of a config file:

```Python

# codec settings
codec = dict(type='RegressionLabel', input_size=(192, 256))                     ## definition ##

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='RLEHead',
        in_channels=2048,
        num_joints=17,
        loss=dict(type='RLELoss', use_target_weight=True),
        decoder=codec),                                                         ## Head ##
    test_cfg=dict(
        flip_test=True,
        shift_coords=True,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),   ## Generate Target ##
    dict(type='PackPoseInputs')
]
test_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
```

## Supported Codecs

Supported codecs are in [$MMPOSE/mmpose/codecs/](https://github.com/open-mmlab/mmpose/tree/dev-1.x/mmpose/codecs). Here is a list:

- [RegressionLabel](#RegressionLabel)
- [IntegralRegressionLabel](#IntegralRegressionLabel)
- [MSRAHeatmap](#MSRAHeatmap)
- [UDPHeatmap](#UDPHeatmap)
- [MegviiHeatmap](#MegviiHeatmap)
- [SPR](#SPR)
- [SimCC](#SimCC)
- [DecoupledHeatmap](#DecoupledHeatmap)
- [ImagePoseLifting](#ImagePoseLifting)
- [VideoPoseLifting](#VideoPoseLifting)
- [MotionBERTLabel](#MotionBERTLabel)

### RegressionLabel

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/regression_label.py#L12)

The `RegressionLabel` codec is used to generate normalized coordinates as the regression targets.

**Input**

- Encoding keypoints from input image space to normalized space.

**Output**

- Decoding normalized coordinates from normalized space to input image space.

Related works:

- [DeepPose](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#deeppose-cvpr-2014)
- [RLE](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#rle-iccv-2021)

### IntegralRegressionLabel

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/integral_regression_label.py)

The `IntegralRegressionLabel` codec is used to generate normalized coordinates as the regression targets.

**Input**

- Encoding keypoints from input image space to normalized space, and generate Gaussian heatmaps as well.

**Output**

- Decoding normalized coordinates from normalized space to input image space.

Related works:

- [IPR](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#ipr-eccv-2018)
- [DSNT](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#dsnt-2018)
- [Debias IPR](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#debias-ipr-iccv-2021)

### MSRAHeatmap

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/msra_heatmap.py)

The `MSRAHeatmap` codec is used to generate Gaussian heatmaps as the targets.

**Input**

- Encoding keypoints from input image space to output space as 2D Gaussian heatmaps.

**Output**

- Decoding 2D Gaussian heatmaps from output space to input image space as coordinates.

Related works:

- [SimpleBaseline2D](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#simplebaseline2d-eccv-2018)
- [CPM](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#cpm-cvpr-2016)
- [HRNet](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#hrnet-cvpr-2019)
- [DARK](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#darkpose-cvpr-2020)

### UDPHeatmap

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/udp_heatmap.py)

The `UDPHeatmap` codec is used to generate Gaussian heatmaps as the targets.

**Input**

- Encoding keypoints from input image space to output space as 2D Gaussian heatmaps.

**Output**

- Decoding 2D Gaussian heatmaps from output space to input image space as coordinates.

Related works:

- [UDP](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#udp-cvpr-2020)

### MegviiHeatmap

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/megvii_heatmap.py)

The `MegviiHeatmap` codec is used to generate Gaussian heatmaps as the targets, which is usually used in Megvii's works.

**Input**

- Encoding keypoints from input image space to output space as 2D Gaussian heatmaps.

**Output**

- Decoding 2D Gaussian heatmaps from output space to input image space as coordinates.

Related works:

- [MSPN](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#mspn-arxiv-2019)
- [RSN](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#rsn-eccv-2020)

### SPR

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/spr.py)

The `SPR` codec is used to generate Gaussian heatmaps of instances' center, and offsets as the targets.

**Input**

- Encoding keypoints from input image space to output space as 2D Gaussian heatmaps and offsets.

**Output**

- Decoding 2D Gaussian heatmaps and offsets from output space to input image space as coordinates.

Related works:

- [DEKR](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#dekr-cvpr-2021)

### SimCC

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/simcc_label.py)

The `SimCC` codec is used to generate 1D Gaussian representations as the targets.

**Input**

- Encoding keypoints from input image space to output space as 1D Gaussian representations.

**Output**

- Decoding 1D Gaussian representations from output space to input image space as coordinates.

Related works:

- [SimCC](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#simcc-eccv-2022)
- [RTMPose](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#rtmpose-arxiv-2023)

### DecoupledHeatmap

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/decoupled_heatmap.py)

The `DecoupledHeatmap` codec is used to generate Gaussian heatmaps as the targets.

**Input**

- Encoding human center points and keypoints from input image space to output space as 2D Gaussian heatmaps.

**Output**

- Decoding 2D Gaussian heatmaps from output space to input image space as coordinates.

Related works:

- [CID](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#cid-cvpr-2022)

### ImagePoseLifting

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/image_pose_lifting.py)

The `ImagePoseLifting` codec is used for image 2D-to-3D pose lifting.

**Input**

- Encoding 2d keypoints from input image space to normalized 3d space.

**Output**

- Decoding 3d keypoints from normalized space to input image space.

Related works:

- [SimpleBaseline3D](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#simplebaseline3d-iccv-2017)

### VideoPoseLifting

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/video_pose_lifting.py)

The `VideoPoseLifting` codec is used for video 2D-to-3D pose lifting.

**Input**

- Encoding 2d keypoints from input image space to normalized 3d space.

**Output**

- Decoding 3d keypoints from normalized space to input image space.

Related works:

- [VideoPose3D](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo_papers/algorithms.html#videopose3d-cvpr-2019)

### MotionBERTLabel

[\[Github\]](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/codecs/motionbert_label.py)

The `MotionBERTLabel` codec is used for video 2D-to-3D pose lifting.

**Input**

- Encoding 2d keypoints from input image space to normalized 3d space.

**Output**

- Decoding 3d keypoints from normalized space to input image space.

Related works:

- [MotionBERT](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo/body_3d_keypoint.html#pose-lift-motionbert-on-h36m)
