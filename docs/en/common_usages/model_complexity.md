# Get Model Params & FLOPs

MMPose provides `tools/analysis_tools/get_flops.py` to get model parameters and FLOPs.

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}] [--cfg-options ${CFG_OPTIONS}]
```

Description of all arguments:

`CONFIG_FILE` : The path of a model config file.

`--shape`: The input shape to the model.

`--input-constructor`: If specified as batch, it will generate a batch tensor to calculate FLOPs.

`--batch-size`：If `--input-constructor` is specified as batch, it will generate a random tensor with shape `(batch_size, 3, **input_shape)` to calculate FLOPs.

`--cfg-options`: If specified, the key-value pair optional `cfg` will be merged into config file.

Example:

```shell
python tools/analysis_tools/get_flops.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
```

We will get the following results:

```text
==============================
Input shape: (1, 3, 256, 192)
Flops: 7.7 GFLOPs
Params: 28.54 M
==============================
```

```{note}
This tool is still experimental and we do not guarantee that the number is absolutely correct. Some operators are not counted into FLOPs like GN and custom operators.
```
