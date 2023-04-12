# How to

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

## Print Entire Config

Officially provided config files inherit multiple config files, which can facilitate management and reduce redundant code. But sometimes we want to know what the default parameter values that are not explicitly written in the configuration file are. MMPose provides `tools/analysis_tools/print_config.py` to print the entire configuration information verbatim.

```shell
python tools/analysis_tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
