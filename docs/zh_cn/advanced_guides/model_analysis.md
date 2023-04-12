# Model Analysis

## 统计模型参数量与计算量

MMPose 提供了 `tools/analysis_tools/get_flops.py` 来统计模型的参数量与计算量。

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}] [--cfg-options ${CFG_OPTIONS}]
```

参数说明：

`CONFIG_FILE` : 模型配置文件的路径。

`--shape`: 模型的输入张量形状。

`--input-constructor`: 如果指定为 `batch`，将会生成一个 `batch tensor` 来计算 FLOPs。

`--batch-size`：如果 `--input-constructor` 指定为 `batch`，将会生成一个随机 `tensor`，形状为 `(batch_size, 3, **input_shape)` 来计算 FLOPs。

`--cfg-options`: 如果指定，可选的 `cfg` 的键值对将会被合并到配置文件中。

示例：

```shell
python tools/analysis_tools/get_flops.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
```

结果如下：

```text
==============================
Input shape: (1, 3, 256, 192)
Flops: 7.7 GFLOPs
Params: 28.54 M
==============================
```

```{note}
目前该工具仍处于实验阶段，我们不能保证统计结果绝对正确，一些算子（比如 GN 或自定义算子）没有被统计到 FLOPs 中。
```
