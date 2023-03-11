# 分析训练日志

MMPose 提供了 `tools/analysis_tools/analyze_logs.py` 来对训练日志进行简单的分析，包括：

- 将日志绘制成损失和精度曲线图
- 统计训练速度

## 绘制损失和精度曲线图

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

## 统计训练速度

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
