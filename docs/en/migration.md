# How to Migrate MMPose 0.x Projects to MMPose 1.0

MMPose 1.0 has been refactored extensively and addressed many legacy issues. Most of the code in MMPose 1.0 will not be compatible with 0.x version.

To try our best to help you migrate your code and model, here are some major changes:

## Data Transformation

### Translation, Rotation and Scaling

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

### Target Generation

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

### Data Normalization

The data normalization operations `NormalizeTensor` and `ToTensor` will be replaced by **DataPreprocessor** module, which will no longer be used as a preprocessing operation, but will be merged as a part of the model forward propagation.

## Compatibility of Models

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

### Heatmap-based Model

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

### RLE-based Model

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
