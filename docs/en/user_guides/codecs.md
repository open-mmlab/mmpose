# Codecs

In the keypoint detection task, depending on the algorithm, it is often necessary to generate targets in different formats, such as normalized coordinates, vectors and heatmaps, etc. Similarly, for the model outputs, a decoding process is required to transform them into coordinates.

In normal open source code, the encoding and decoding processes are usually scattered across many files. This makes the pair of processes, which are mutually inverse, less intuitive and unified.

MMPose propose the `Codec` to integrate the `encoder` and `decoder` together, to make them modular and user friendly.

Here is a diagram to show where the Codec is:

![img](https://aicarrier.feishu.cn/space/api/box/stream/download/asynccode/?code=MDUwYjJkZjJiNDMzYzg0YTNkODUyOWJmM2UzMzk5YTdfYzdyVG4wTTh1NnVmMTZyMHZzN3VjUW9jUVlnM2ZKT3RfVG9rZW46Ym94Y25lblBtVEl3RzdENTNhUVpGT05WTGhjXzE2NjE1MDYxNTQ6MTY2MTUwOTc1NF9WNA)

A typical codec consists of two parts:

- Encoder

- Decoder

### Encoder

The encoder transforms the coordinates in the input image space into the needed target format:

- Normalized Coordinates

- One-dimensional Vectors

- Gaussian Heatmaps

Here is the definition of the encoder：

```Python
@abstractmethod
def encode(self,
           keypoints: np.ndarray,
           keypoints_visible: Optional[np.ndarray] = None) -> Any:
    """Encode keypoints.
    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
    Args:
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibility in shape
            (N, K, D)
    """
```

### Decoder

The decoder transforms the model outputs into coordinates in the input image space, which is the opposite processing of the encoder.

Here is the definition of the decoder:

```Python
@abstractmethod
def decode(self, encoded: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Decode keypoints.
    Args:
        encoded (any): Encoded keypoint representation using the codec
    Returns:
        tuple:
        - keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        - keypoints_visible (np.ndarray): Keypoint visibility in shape
            (N, K, D)
    """
```

By default, the `decode` method only performs decoding on a single instance. You can also implement the `batch_decode` method to boost the decoding process.

## Common Usage

The example below shows how to use a codec in your config:

- Definition

- Generate Target

- Head

### Definition

Take the Regression-based method to generate normalized coordinates as an example, you can define a `codec` in your config as follows:

```Python
codec = dict(type='RegressionLabel', input_size=(192, 256))
```

### Generate Target

In pipelines, A codec should be passed into `GenerateTarget` to work as the `encoder`:

```Python
dict(type='GenerateTarget', target_type='keypoint_label', encoder=codec)
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
