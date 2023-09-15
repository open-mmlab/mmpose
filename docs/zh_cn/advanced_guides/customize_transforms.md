# 自定义数据变换和数据增强

### 数据变换

在**OpenMMLab**算法库中，数据集的构建和数据的准备是相互解耦的，通常，数据集的构建只对数据集进行解析，记录每个样本的基本信息，而数据的准备则是通过一系列的数据变换，根据样本的基本信息进行数据加载、预处理、格式化等操作。

### 数据变换的使用

**MMPose**中的`数据变换`和`数据增强`类定义在[$MMPose/datasets/transforms](https://github.com/open-mmlab/mmpose/tree/dev-1.x/mmpose/datasets/transforms)目录中，对应的文件结构如下:

```txt
mmpose
|----datasets
    |----transforms
        |----bottomup_transforms    # 自底向上
        |----common_transforms      # 常用变换
        |----converting             # 关键点转换
        |----formatting             # 输入数据格式化
        |----loading                # 原始数据加载
        |----pose3d_transforms      # 三维变换
        |----topdown_transforms     # 自顶向下
```

在**MMPose**中，**数据增强**和**数据变换**是使用者经常需要考虑的一个阶段，可参考如下流程进行相关阶段的设计：

[![](https://mermaid.ink/img/pako:eNp9UT1LA0EQ_SvH1knhXXeFhYpfWIQklVwz3s7mFvd2jr09JIaAjQQRRLAU0Ua0t5X8m4v-DHcThPOIqfbNe2_fMDMTlhJHFjOh6CLNwNjgpJ_oICirs5GBIguGBnQpyOTlluf3lSz8ewhK7BAfe9wnC1aS9niQSWGXJJbyEj3aJUXmWFpLxpeo-T8NQs8foEYDFodgRmg3f4g834P0vEclHumiavrru-f67bZ-nH_dzIJud7s9yUpfvMwWH-_r9Ea5lL_nD_X16ypvg_507yLrq09vaVGtLuHflLAlR42EdUNErMNyNDlI7u438e6E2QxzTFjsIEcBlbIJS_TUWaGyNBjrlMXWVNhhVcHdlvckuKXmLBagSscWoE-JfuvpD2uI1Wk?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNp9UT1LA0EQ_SvH1knhXXeFhYpfWIQklVwz3s7mFvd2jr09JIaAjQQRRLAU0Ua0t5X8m4v-DHcThPOIqfbNe2_fMDMTlhJHFjOh6CLNwNjgpJ_oICirs5GBIguGBnQpyOTlluf3lSz8ewhK7BAfe9wnC1aS9niQSWGXJJbyEj3aJUXmWFpLxpeo-T8NQs8foEYDFodgRmg3f4g834P0vEclHumiavrru-f67bZ-nH_dzIJud7s9yUpfvMwWH-_r9Ea5lL_nD_X16ypvg_507yLrq09vaVGtLuHflLAlR42EdUNErMNyNDlI7u438e6E2QxzTFjsIEcBlbIJS_TUWaGyNBjrlMXWVNhhVcHdlvckuKXmLBagSscWoE-JfuvpD2uI1Wk)

[common_transforms](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/common_transforms.py)组件提供了常用的`RandomFlip`,`RandomHalfBody`数据增强算法。

- `Top-Down`方法中`Shift`,`Rotate`,`Resize`等操作体现为[RandomBBoxTransform](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/common_transforms.py#L435)方法。
- `Buttom-Up`算法中体现为[BottomupResize](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/bottomup_transforms.py#L327)方法。
- `pose-3d`则为[RandomFlipAroundRoot](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/pose3d_transforms.py#L13)方法。

**MMPose**对于`Top-Down`、`Buttom-Up`，`pose-3d`都提供了对应的数据变换接口。通过采用仿射变换，将图像和坐标标注从`原始图片空间`变换到`输入图片空间`。

- `Top-Down`方法中体现为[TopdownAffine](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/topdown_transforms.py#L14)。
- `Buttom-Up`方法体现为[BottomupRandomAffine](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/bottomup_transforms.py#L134)。

以[RandomFlip](https://github.com/open-mmlab/mmpose/blob/main/mmpose/datasets/transforms/common_transforms.py)为例，该方法随机的对`原始图片`进行变换，并转换为`输入图像`或`中间图像`。要定义一个数据变换的过程，需要继承[BaseTransform](https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/base.py)类，并进行`TRANSFORM`注册：

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

**输入**：

- `prob`指定了在水平，垂直，斜向等变换的概率，是一个范围在\[0,1\]之间的浮点数`list`。
- `direction`指定了数据变换的方向：
  - `horizontal`水平变换
  - `vertical`垂直变换
  - `diagonal`对角变换

**输出**：

- 输出一个经过**数据变换**后的`dict`数据

`RandomFlip`的[transform](https://github.com/open-mmlab/mmpose/blob/main/mmpose/datasets/transforms/common_transforms.py#L187)实现了对输入图像的以给定的`prob`概率进行水平、垂直或是对角方向的数据翻转，并返回输出图像。

以下是使用`对角翻转变换`的一个简单示例：

```python
from mmpose.datasets.transforms import LoadImage, RandomFlip
import mmcv

# 从路径中加载原始图片
results = dict(
  img_path='data/test/multi-person.jpeg'
  )
transform = LoadImage()
results = transform(results)
# 此时，加载的原始图片是一个包含以下属性的`dict`:
# - `img_path`: 图片的绝对路径
# - `img`: 图片的像素点
# - `img_shape`: 图片的形状
# - `ori_shape`: 图片的原始形状

# 对原始图像进行对角翻转变换
transform = RandomFlip(prob=1., direction='diagonal')
results = transform(results)
# 此时，加载的原始图片是一个包含以下属性的`dict`:
# - `img_path`: 图片的绝对路径
# - `img`: 图片的像素点
# - `img_shape`: 图片的形状
# - `ori_shape`: 图片的原始形状
# - `flip`: 图片是否进行翻转变换
# - `flip_direction`: 图片进行翻转变换的方向

# 取出经过翻转变换后的图片
mmcv.imshow(results['img'])
```

更多有关自定义数据变换和增强的使用方法，可以参考[$MMPose/test/test_datasets/test_transforms/test_common_transforms](https://github.com/open-mmlab/mmpose/blob/main/tests/test_datasets/test_transforms/test_common_transforms.py#L59)等。

#### RandomHalfBody

[RandomHalfBody](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/common_transforms.py#L263)**数据增强**算法概率的进行上半身或下半身的**数据变换**。
**输入**：

- `min_total_keypoints`最小总关键点数
- `min_half_keypoints`最小半身关键点数
- `padding`bbox的填充比例
- `prob`在关键点数目符合要求下，接受半身变换的概率

**输出**：

- 输出一个经过**数据变换**后的`dict`数据

#### TopdownAffine

[TopdownAffine](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/datasets/transforms/topdown_transforms.py#L14)**数据变换**算法通过仿射变换将`原始图片`变换为`输入图片`。

**输入**：

- `input_size`bbox区域将会被裁剪和修正到的\[w,h\]大小
- `use_udp`是否使用公正的数据过程[UDP](https://arxiv.org/abs/1911.07524)

**输出**：

- 输出一个经过**数据变换**后的`dict`数据

### 在流水线中使用数据增强和变换

配置文件中的**数据增强**和**数据变换**过程可以是如下示例：

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

示例中的流水线对输入数据进行**数据增强**，进行随机的水平增强和半身增强，
并进行`Top-Down`的`Shift`、`Rotate`、`Resize`操作，通过`TopdownAffine`操作实现仿射变换，变换至`输入图片空间`。
