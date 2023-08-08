# Customize Data Transformation and Augmentation

## DATA TRANSFORM

In the OpenMMLab algorithm library, the construction of the dataset and the preparation of the data are decoupled from each other. Usually, the construction of the dataset only analyzes the dataset and records the basic information of each sample, while the preparation of the data is through a series of According to the basic information of the sample, perform data loading, preprocessing, formatting and other operations.

### The use of data transformation

The **data transformation** and **data augmentation** classes in **MMPose** are defined in the [$MMPose/datasets/transforms](https://github.com/open-mmlab/mmpose/tree/dev-1.x/mmpose/datasets/transforms) directory, and the corresponding file structure is as follows:

```txt
mmpose
|----datasets
    |----transforms
        |----bottomup_transforms    # Button-Up transforms
        |----common_transforms      # Common Transforms
        |----converting             # Keypoint converting
        |----formatting             # Input data formatting
        |----loading                # Raw data loading
        |----pose3d_transforms      # Pose3d-transforms
        |----topdown_transforms     # Top-Down transforms
```

In **MMPose**, **data augmentation** and **data transformation** is a stage that users often need to consider. You can refer to the following process to design related stages:

[![](https://mermaid.ink/img/pako:eNp9UbFOwzAQ_ZXIczuQbBkYKAKKOlRpJ5TlGp8TC9sX2WdVpeq_Y0cClahl8rv3nt_d2WfRkURRC2Xo2A3gudg0rSuKEA-9h3Eo9h5cUORteMj8i9FjPt_AqCeSp4wbYmBNLuPdoBVPJAb9hRmtyJB_18zkc4lO3mlQZv4VHXpg3IPvkf-_UGV-C93nlgKu3Riv_Q0c1xZ6LJbLx_kWSdvAAc0t7aqc5Cl3Srqrroi81C5NHbJnzs26lH9zyplc_UbcGr8SC2HRW9Ay_do5e1vBA1psRZ2gRAXRcCtad0lWiEy7k-tEzT7iQsRRpomeNaSntKJWYEJiR3AfRD_15RuTF7md?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNp9UbFOwzAQ_ZXIczuQbBkYKAKKOlRpJ5TlGp8TC9sX2WdVpeq_Y0cClahl8rv3nt_d2WfRkURRC2Xo2A3gudg0rSuKEA-9h3Eo9h5cUORteMj8i9FjPt_AqCeSp4wbYmBNLuPdoBVPJAb9hRmtyJB_18zkc4lO3mlQZv4VHXpg3IPvkf-_UGV-C93nlgKu3Riv_Q0c1xZ6LJbLx_kWSdvAAc0t7aqc5Cl3Srqrroi81C5NHbJnzs26lH9zyplc_UbcGr8SC2HRW9Ay_do5e1vBA1psRZ2gRAXRcCtad0lWiEy7k-tEzT7iQsRRpomeNaSntKJWYEJiR3AfRD_15RuTF7md)

The `common_transforms` component provides commonly used `RandomFlip`, `RandomHalfBody` **data augmentation**.

- Operations such as `Shift`, `Rotate`, and `Resize` in the `Top-Down` method are reflected in the [RandomBBoxTransform](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/common_transforms.py#L435) method.
- The [BottomupResize](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/bottomup_transforms.py#L327) method is embodied in the `Buttom-Up` algorithm.
- `pose-3d` is the [RandomFlipAroundRoot](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/pose3d_transforms.py#L13) method.

**MMPose** provides corresponding data conversion interfaces for `Top-Down`, `Button-Up`, and `pose-3d`. Transform the image and coordinate labels from the `original_image_space` to the `input_image_space` by using an affine transformation.

- The `Top-Down` method is manifested as [TopdownAffine](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/topdown_transforms.py#L14).
- The `Bottom-Up` method is embodied as [BottomupRandomAffine](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/bottomup_transforms.py#L134).

Taking `RandomFlip` as an example, this method randomly transforms the `original_image` and converts it into an `input_image` or an `intermediate_image`. To define a data transformation process, you need to inherit the [BaseTransform](https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/base.py) class and register with `TRANSFORM`:

```python
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
      """Randomly flip the image, bbox and keypoints.

    Required Keys:

        - img
        - img_shape
        - flip_indices
        - input_size (optional)
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)
        - img_mask (optional)

    Modified Keys:

        - img
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)
        - img_mask (optional)

    Added Keys:

        - flip
        - flip_direction

    Args:
        prob (float | list[float]): The flipping probability. If a list is
            given, the argument `direction` should be a list with the same
            length. And each element in `prob` indicates the flipping
            probability of the corresponding one in ``direction``. Defaults
            to 0.5
        direction (str | list[str]): The flipping direction. Options are
            ``'horizontal'``, ``'vertical'`` and ``'diagonal'``. If a list is
            is given, each data sample's flipping direction will be sampled
            from a distribution determined by the argument ``prob``. Defaults
            to ``'horizontal'``.
    """
    def __init__(self,
                prob: Union[float, List[float]] = 0.5,
                direction: Union[str, List[str]] = 'horizontal') -> None:
      if isinstance(prob, list):
          assert is_list_of(prob, float)
          assert 0 <= sum(prob) <= 1
      elif isinstance(prob, float):
          assert 0 <= prob <= 1
      else:
          raise ValueError(f'probs must be float or list of float, but \
                            got `{type(prob)}`.')
      self.prob = prob

      valid_directions = ['horizontal', 'vertical', 'diagonal']
      if isinstance(direction, str):
          assert direction in valid_directions
      elif isinstance(direction, list):
          assert is_list_of(direction, str)
          assert set(direction).issubset(set(valid_directions))
      else:
          raise ValueError(f'direction must be either str or list of str, \
                              but got `{type(direction)}`.')
      self.direction = direction

      if isinstance(prob, list):
          assert len(prob) == len(self.direction)
```

**Input**:

- `prob` specifies the probability of transformation in horizontal, vertical, diagonal, etc., and is a `list` of floating-point numbers in the range \[0,1\].
- `direction` specifies the direction of data transformation:
  - `horizontal`
  - `vertical`
  - `diagonal`

**Output**:

- Return a `dict` data after data transformation.

Here is a simple example of using `diagonal  RandomFlip`：

```python
from mmpose.datasets.transforms import LoadImage, RandomFlip
import mmcv

# Load the original image from the path
results = dict(
  img_path='data/test/multi-person.jpeg'
  )
transform = LoadImage()
results = transform(results)
# At this point, the original image loaded is a `dict`
# that contains the following attributes`:
# - `img_path`: Absolute path of image
# - `img`: Pixel points of the image
# - `img_shape`: The shape of the image
# - `ori_shape`: The original shape of the image

# Perform diagonal flip transformation on the original image
transform = RandomFlip(prob=1., direction='diagonal')
results = transform(results)
# At this point, the original image loaded is a `dict`
# that contains the following attributes`:
# - `img_path`: Absolute path of image
# - `img`: Pixel points of the image
# - `img_shape`: The shape of the image
# - `ori_shape`: The original shape of the image
# - `flip`: Is the image flipped and transformed
# - `flip_direction`: The direction in which
# the image is flipped and transformed

# Get the image after flipping and transformation
mmcv.imshow(results['img'])
```

For more information on using custom data transformations and enhancements, please refer to [$MMPose/test/test_datasets/test_transforms/test_common_transforms](https://github.com/open-mmlab/mmpose/blob/main/tests/test_datasets/test_transforms/test_common_transforms.py#L59)。

#### RandomHalfBody

The [RandomHalfBody](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/common_transforms.py#L263) **data augmentation** algorithm probabilistically transforms the data of the upper or lower body.

**Input**:

- `min_total_keypoints` minimum total keypoints
- `min_half_keypoints` minimum half-body keypoints
- `padding` The filling ratio of the bbox
- `prob` accepts the probability of half-body transformation when the number of key points meets the requirements

**Output**:

- Return a `dict` data after data transformation.

#### Topdown Affine

The [TopdownAffine](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/topdown_transforms.py#L14) data transformation algorithm transforms the `original image` into an `input image` through affine transformation

- `input_size` The bbox area will be cropped and corrected to the \[w,h\] size
- `use_udp` whether to use fair data process [UDP](https://arxiv.org/abs/1911.07524).

**Output**:

- Return a `dict` data after data transformation.

### Using Data Augmentation and Transformation in the Pipeline

The **data augmentation** and **data transformation** process in the configuration file can be the following example:

```python
train_pipeline_stage2 = [
    ...
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(
         type='TopdownAffine',
         input_size=codec['input_size']),
    ...
]
```

The pipeline in the example performs **data enhancement** on the `input data`, performs random horizontal transformation and half-body transformation, and performs `Top-Down` `Shift`, `Rotate`, and `Resize` operations, and implements affine transformation through `TopdownAffine` operations to transform to the `input_image_space`.
