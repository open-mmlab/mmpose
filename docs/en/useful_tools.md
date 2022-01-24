# Useful Tools

Apart from training/testing scripts, We provide lots of useful tools under the `tools/` directory.

<!-- TOC -->

- [Log Analysis](#log-analysis)
- [Model Complexity (experimental)](#model-complexity-experimental)
- [Model Conversion](#model-conversion)
  - [MMPose model to ONNX (experimental)](#mmpose-model-to-onnx-experimental)
  - [Prepare a model for publishing](#prepare-a-model-for-publishing)
- [Model Serving](#model-serving)
- [Miscellaneous](#miscellaneous)
  - [Evaluating a metric](#evaluating-a-metric)
  - [Print the entire config](#print-the-entire-config)

<!-- TOC -->

## Log Analysis

`tools/analysis/analyze_logs.py` plots loss/pose acc curves given a training log file. Run `pip install seaborn` first to install the dependency.

![acc_curve_image](imgs/acc_curve.png)

```shell
python tools/analysis/analyze_logs.py plot_curve ${JSON_LOGS} [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the mse loss of some run.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log.json --keys loss --legend loss
  ```

- Plot the acc of some run, and save the figure to a pdf.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log.json --keys acc_pose --out results.pdf
  ```

- Compare the acc of two runs in the same figure.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log1.json log2.json --keys acc_pose --legend run1 run2
  ```

You can also compute the average training speed.

```shell
python tools/analysis/analyze_logs.py cal_train_time ${JSON_LOGS} [--include-outliers]
```

- Compute the average training speed for a config file

  ```shell
  python tools/analysis/analyze_logs.py cal_train_time log.json
  ```

  The output is expected to be like the following.

  ```text
  -----Analyze train time of log.json-----
  slowest epoch 114, average time is 0.9662
  fastest epoch 16, average time is 0.7532
  time std over epochs is 0.0426
  average iter time: 0.8406 s/iter
  ```

## Model Complexity (Experimental)

`/tools/analysis/get_flops.py` is a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/analysis/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

We will get the result like this

```text

==============================
Input shape: (1, 3, 256, 192)
Flops: 8.9 GMac
Params: 28.04 M
==============================
```

```{note}
This tool is still experimental and we do not guarantee that the number is absolutely correct.
```

You may use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 340, 256) for 2D recognizer, (1, 3, 32, 340, 256) for 3D recognizer.
(2) Some operators are not counted into FLOPs like GN and custom operators. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.

## Model Conversion

### MMPose model to ONNX (experimental)

`/tools/deployment/pytorch2onnx.py` is a script to convert model to [ONNX](https://github.com/onnx/onnx) format.
It also supports comparing the output results between Pytorch and ONNX model for verification.
Run `pip install onnx onnxruntime` first to install the dependency.

```shell
python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --shape $SHAPE --verify
```

### Prepare a model for publishing

`tools/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to:

(1) convert model weights to CPU tensors.
(2) delete the optimizer states.
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/hrnet_w32_coco_256x192/latest.pth hrnet_w32_coco_256x192
```

The final output filename will be `hrnet_w32_coco_256x192-{hash id}_{time_stamp}.pth`.

## Model Serving

MMPose supports model serving with [`TorchServe`](https://pytorch.org/serve/). You can serve an MMPose model via following steps:

### 1. Install TorchServe

Please follow the official installation guide of TorchServe: https://github.com/pytorch/serve#install-torchserve-and-torch-model-archiver

### 2. Convert model from MMPose to TorchServe

```shell
python tools/deployment/mmpose2torchserve.py \
  ${CONFIG_FILE} ${CHECKPOINT_FILE} \
  --output-folder ${MODEL_STORE} \
  --model-name ${MODEL_NAME}
```

**Note**: ${MODEL_STORE} needs to be an absolute path to a folder.

A model file `${MODEL_NAME}.mar` will be generated and placed in the `${MODEL_STORE}` folder.

### 3. Deploy model serving

We introduce following 2 approaches to deploying the model serving.

#### Use TorchServe API

```shell
torchserve --start \
  --model-store ${MODEL_STORE} \
  --models ${MODEL_PATH1} [${MODEL_NAME}=${MODEL_PATH2} ... ]
```

Example:

```shell
# serve one model
torchserve --start --model-store /models --models hrnet=hrnet.mar

# serve all models in model-store
torchserve --start --model-store /models --models all
```

After executing the `torchserve` command above, TorchServe runse on your host, listening for inference requests. Check the [official docs](https://github.com/pytorch/serve/blob/master/docs/server.md) for more information.

#### Use `mmpose-serve` docker image

**Build `mmpose-serve` docker image:**

```shell
docker build -t mmpose-serve:latest docker/serve/
```

**Run `mmpose-serve`:**

Check the official docs for [running TorchServe with docker](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

In order to run in GPU, you need to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). You can omit the `--gpus` argument in order to run in CPU.

Example:

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmpose-serve:latest
```

[Read the docs](https://github.com/pytorch/serve/blob/072f5d088cce9bb64b2a18af065886c9b01b317b/docs/rest_api.md/) about the Inference (8080), Management (8081) and Metrics (8082) APis

### 4. Test deployment

You can use `tools/deployment/test_torchserver.py` to test the model serving. It will compare and visualize the result of torchserver and pytorch.

```shell
python tools/deployment/test_torchserver.py ${IMAGE_PAHT} ${CONFIG_PATH} ${CHECKPOINT_PATH} ${MODEL_NAME} --out-dir ${OUT_DIR}
```

Example:

```shell
python tools/deployment/test_torchserver.py \
  ls tests/data/coco/000000000785.jpg \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
  https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
  hrnet \
  --out-dir vis_results
```

## Miscellaneous

### Print the entire config

`tools/analysis/print_config.py` prints the whole config verbatim, expanding all its imports.

```shell
python tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
