# MMPose 0.X 兼容性说明

MMPose 1.0 经过了大规模重构并解决了许多遗留问题，对于 0.x 版本的大部分代码 MMPose 1.0 将不兼容。

## 数据变换

### 平移、旋转和缩放

旧版的数据变换方法 `TopDownRandomShiftBboxCenter` 和 `TopDownGetRandomScaleRotation`，将被合并为 `RandomBBoxTransform`：

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

### 标签生成

旧版用于训练标签生成的方法 `TopDownGenerateTarget` 、`TopDownGenerateTargetRegression`、`BottomUpGenerateHeatmapTarget`、`BottomUpGenerateTarget` 等将被合并为 `GenerateTarget`，而实际的生成方法由[编解码器](./user_guides/codecs.md) 提供：

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

    Added Keys:

        - The keys of the encoded items from the codec will be updated into
            the results, e.g. ``'heatmaps'`` or ``'keypoint_weights'``. See
            the specific codec for more details.

    Args:
        encoder (dict | list[dict]): The codec config for keypoint encoding.
            Both single encoder and multiple encoders (given as a list) are
            supported
        multilevel (bool): Determine the method to handle multiple encoders.
            If ``multilevel==True``, generate multilevel targets from a group
            of encoders of the same type (e.g. multiple :class:`MSRAHeatmap`
            encoders with different sigma values); If ``multilevel==False``,
            generate combined targets from a group of different encoders. This
            argument will have no effect in case of single encoder. Defaults
            to ``False``
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
    """

    def __init__(self,
                 encoder: MultiConfig,
                 multilevel: bool = False,
                 use_dataset_keypoint_weights: bool = False) -> None:
```

### 数据归一化

旧版的数据归一化操作 `NormalizeTensor` 和 `ToTensor` 方法将由 **DataPreprocessor** 模块替代，不再作为流水线的一部分，而是作为模块加入到模型前向传播中。

## 模型兼容

我们对 model zoo 提供的模型权重进行了兼容性处理，确保相同的模型权重测试精度能够与 0.x 版本保持同等水平，但由于在这两个版本中存在大量处理细节的差异，推理结果可能会产生轻微的不同（精度误差小于 0.05%）。

对于使用 0.x 版本训练保存的模型权重，我们在预测头中提供了 `_load_state_dict_pre_hook()` 方法来将旧版的权重字典替换为新版，如果你希望将在旧版上开发的模型兼容到新版，可以参考我们的实现。

```Python
@MODELS.register_module()
class YourHead(BaseHead):
def __init__(self):

    ## omitted

    # Register the hook to automatically convert old version state dicts
    self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
```

### Heatmap-based 方法

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

### RLE-based 方法

对于基于 RLE 的模型，由于新版的 `loss` 模块更名为 `loss_module`，且 flow 模型归属在 `loss` 模块下，因此需要对权重字典中 `loss` 字段进行更改：

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
