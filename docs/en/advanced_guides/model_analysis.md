# Model Analysis

## Get Model Params & FLOPs

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

## Log Analysis

MMPose provides `tools/analysis_tools/analyze_logs.py` to analyze the training log. The log file can be either a json file or a text file. The json file is recommended, because it is more convenient to parse and visualize.

Currently, the following functions are supported:

- Plot loss/accuracy curves
- Calculate training time

### Plot Loss/Accuracy Curves

The function depends on `seaborn`, please install it first by running `pip install seaborn`.

![log_curve](https://user-images.githubusercontent.com/87690686/188538215-5d985aaa-59f8-44cf-b6f9-10890d599e9c.png)

```shell
python tools/analysis_tools/analyze_logs.py plot_curve ${JSON_LOGS} [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot loss curve

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_kpt --legend loss_kpt
  ```

- Plot accuracy curve and export to PDF file

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys acc_pose --out results.pdf
  ```

- Plot multiple log files on the same figure

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys loss_kpt --legend run1 run2 --title loss_kpt --out loss_kpt.png
  ```

### Calculate Training Time

```shell
python tools/analysis_tools/analyze_logs.py cal_train_time ${JSON_LOGS} [--include-outliers]
```

Examples:

```shell
python tools/analysis_tools/analyze_logs.py cal_train_time log.json
```

The result is as follows:

```text
-----Analyze train time of hrnet_w32_256x192.json-----
slowest epoch 56, average time is 0.6924
fastest epoch 1, average time is 0.6502
time std over epochs is 0.0085
average iter time: 0.6688 s/iter
```
