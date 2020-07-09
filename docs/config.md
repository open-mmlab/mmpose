# Config System
We incorporate modular and inheritance (TODO) design into our config system, which is convenient to conduct various experiments.

## Config File Naming Style

We follow the style below to name config files. Contributors are advised to follow the same style.

```
configs/{method}/{model}/{model}_[model setting]_{backbone}_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{method}`: method type, e.g. `topdown`, `bottomup`, etc.
- `{model}`: model type, e.g. `sbl`, `hr`, etc.
- `[model setting]`: specific setting for some models.
- `{backbone}`: backbone type, e.g. `resnet50` (ResNet-50), etc.
- `[misc]`: miscellaneous setting/plugins of model, e.g. `dense`, `2d`, `3d`, etc.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8x64` is used by default.
- `{schedule}`: training schedule, e.g. `100e`, `210e`, etc. `210e` denotes training 210 epochs, which is used by default.
- `{dataset}`: dataset name, e.g. `coco`, etc.
