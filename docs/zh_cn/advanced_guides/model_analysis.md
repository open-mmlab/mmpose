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

## 分析训练日志

MMPose 提供了 `tools/analysis_tools/analyze_logs.py` 来对训练日志进行简单的分析，包括：

- 将日志绘制成损失和精度曲线图
- 统计训练速度

### 绘制损失和精度曲线图

该功能依赖于 `seaborn`，请先运行 `pip install seaborn` 安装依赖包。

![log_curve](https://user-images.githubusercontent.com/87690686/188538215-5d985aaa-59f8-44cf-b6f9-10890d599e9c.png)

```shell
python tools/analysis_tools/analyze_logs.py plot_curve ${JSON_LOGS} [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

示例：

- 绘制损失曲线

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_kpt --legend loss_kpt
  ```

- 绘制精度曲线并导出为 PDF 文件

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys acc_pose --out results.pdf
  ```

- 将多个日志文件绘制在同一张图上

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys loss_kpt --legend run1 run2 --title loss_kpt --out loss_kpt.png
  ```

### 统计训练速度

```shell
python tools/analysis_tools/analyze_logs.py cal_train_time ${JSON_LOGS} [--include-outliers]
```

示例：

```shell
python tools/analysis_tools/analyze_logs.py cal_train_time log.json
```

结果如下：

```text
-----Analyze train time of hrnet_w32_256x192.json-----
slowest epoch 56, average time is 0.6924
fastest epoch 1, average time is 0.6502
time std over epochs is 0.0085
average iter time: 0.6688 s/iter
```
