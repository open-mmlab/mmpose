# Config System

We use python files as configs. You can find all the provided configs under `$MMPose/configs`.

<!-- TOC -->

- [Config File Naming Convention](#config-file-naming-convention)
- [Config File Structure](#config-file-structure)
  - [Config System for Action localization](#config-system-for-action-localization)
  - [Config System for Action Recognition](#config-system-for-action-recognition)
- [FAQ](#faq)
  - [Use intermediate variables in configs](#use-intermediate-variables-in-configs)

<!-- TOC -->

## Config File Naming Style

We follow the style below to name config files. Contributors are advised to follow the same style.

```
configs/{task}/{model}/{dataset}/{backbone}_[model_setting]_{dataset}_{input_size}_[technique].py
```

`{xxx}` is required field and `[yyy]` is optional.

- `{task}`: method type, e.g. `top_down`, `bottom_up`, `hand`, `mesh`, etc.
- `{model}`: model type, e.g. `hrnet`, `darkpose`, etc.
- `{dataset}`: dataset name, e.g. `coco`, etc.
- `{backbone}`: backbone type, e.g. `res50` (ResNet-50), etc.
- `[model setting]`: specific setting for some models.
- `[misc]`: miscellaneous setting/plugins of model, e.g. `video`, etc.
- `{input_size}`: input size of the model.
- `[technique]`: some specific techniques or tricks to use, e.g. `dark`, `udp`.
