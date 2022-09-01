# 编解码器

在关键点检测任务中，根据算法的不同，需要利用标注信息，生成不同格式的训练目标，比如归一化的坐标值、一维向量、高斯热图等。同样的，对于模型输出的结果，也需要经过处理转换成标注信息格式。我们一般将标注信息到训练目标的处理过程称为编码，模型输出到标注信息的处理过程称为解码。

在目前的开源项目代码中，编码和解码过程往往散落在不同模块里，使得这一对原本互逆的处理过程不够直观和统一，增加了用户的阅读成本。

MMPose使用Codec来将关键点数据的编码器和解码器集成到一起，以增加代码的友好度和复用性。

Codec在工作流程中所处的位置如下所示：

![codec-cn](https://user-images.githubusercontent.com/13503330/187829784-4d5939de-97d7-43cc-b934-c6d17c02d589.png)

一个编解码器（Codec）主要包含两个部分：

- 编码器

- 解码器

### 编码器

编码器主要负责将处于输入图片尺度的坐标值，编码为模型训练所需要的目标格式，主要包括：

- 归一化的坐标值：用于Regression-based方法

- 一维向量：用于SimCC-based方法

- 高斯热图：用于Heatmap-based方法

以Regression-based方法的编码器为例：

```Python
@abstractmethod
def encode(
    self,
    keypoints: np.ndarray,
    keypoints_visible: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Encoding keypoints from input image space to normalized space.

    Args:
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)

    Returns:
        tuple:
        - reg_labels (np.ndarray): The normalized regression labels in
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

    reg_labels = (keypoints / np.array([w, h])).astype(np.float32)
    keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

    return reg_labels, keypoint_weights
```

### 解码器

解码器主要负责将模型的输出解码为输入图片尺度的坐标值，处理过程与编码器相反。

以Regression-based方法的解码器为例：

```Python
def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode keypoint coordinates from normalized space to input image
    space.

    Args:
        encoded (np.ndarray): Coordinates in shape (N, K, D)

    Returns:
        tuple:
        - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
        - socres (np.ndarray): The keypoint scores in shape (N, K).
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

默认情况下，`decode()` 方法只提供单个目标数据的解码过程，你也可以通过`batch_decode()` 来实现批量解码提升执行效率。

## 常见用法

在MMPose配置文件中，主要有三个地方用到Codec：

- 定义

- 生成训练目标

- 模型头部

### 定义

以回归方法生成归一化的坐标值为例，在配置文件中，我们通过如下方式定义Codec：

```Python
codec = dict(type='RegressionLabel', input_size=(192, 256))
```

### 生成训练目标

在数据处理阶段生成训练目标时，需要传入Codec作为编码器：

```Python
dict(type='GenerateTarget', target_type='keypoint_label', encoder=codec)
```

### 模型头部

在MMPose中，我们在模型头部对模型的输出进行解码，需要传入Codec作为解码器：

```Python
head=dict(
    type='RLEHead',
    in_channels=2048,
    num_joints=17,
    loss=dict(type='RLELoss', use_target_weight=True),
    decoder=codec
)
```
