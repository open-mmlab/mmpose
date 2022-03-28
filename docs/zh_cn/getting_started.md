# 基础教程

本文档提供 MMPose 的基础使用教程。请先参阅 [安装指南](install.md)，进行 MMPose 的安装。

<!-- TOC -->

- [准备数据集](#准备数据集)
- [使用预训练模型进行推理](#使用预训练模型进行推理)
  - [测试某个数据集](#测试某个数据集)
  - [运行演示](#运行演示)
- [如何训练模型](#如何训练模型)
  - [使用单个 GPU 训练](#使用单个-GPU-训练)
  - [使用 CPU 训练](#使用-CPU-训练)
  - [使用多个 GPU 训练](#使用多个-GPU-训练)
  - [使用多台机器训练](#使用多台机器训练)
  - [使用单台机器启动多个任务](#使用单台机器启动多个任务)
- [基准测试](#基准测试)
- [进阶教程](#进阶教程)

<!-- TOC -->

## 准备数据集

MMPose 支持各种不同的任务。请根据需要，查阅对应的数据集准备教程。

- [2D 人体关键点检测](/docs/zh_cn/tasks/2d_body_keypoint.md)
- [3D 人体关键点检测](/docs/zh_cn/tasks/3d_body_keypoint.md)
- [3D 人体形状恢复](/docs/zh_cn/tasks/3d_body_mesh.md)
- [2D 人手关键点检测](/docs/zh_cn/tasks/2d_hand_keypoint.md)
- [3D 人手关键点检测](/docs/zh_cn/tasks/3d_hand_keypoint.md)
- [2D 人脸关键点检测](/docs/zh_cn/tasks/2d_face_keypoint.md)
- [2D 全身人体关键点检测](/docs/zh_cn/tasks/2d_wholebody_keypoint.md)
- [2D 服饰关键点检测](/docs/zh_cn/tasks/2d_fashion_landmark.md)
- [2D 动物关键点检测](/docs/zh_cn/tasks/2d_animal_keypoint.md)

## 使用预训练模型进行推理

MMPose 提供了一些测试脚本用于测试数据集上的指标（如 COCO, MPII 等），
并提供了一些高级 API，使您可以轻松使用 MMPose。

### 测试某个数据集

- [x] 单 GPU 测试
- [x] CPU 测试
- [x] 单节点多 GPU 测试
- [x] 多节点测试

用户可使用以下命令测试数据集

```shell
# 单 GPU 测试
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--fuse-conv-bn] \
    [--eval ${EVAL_METRICS}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--cfg-options ${CFG_OPTIONS}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]

# CPU 测试：禁用 GPU 并运行测试脚本
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}]

# 多 GPU 测试
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] \
    [--gpu-collect] [--tmpdir ${TMPDIR}] [--options ${OPTIONS}] [--average-clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]
```

此处的 `CHECKPOINT_FILE` 可以是本地的模型权重文件的路径，也可以是模型的下载链接。

可选参数:

- `RESULT_FILE`：输出结果文件名。如果没有被指定，则不会保存测试结果。
- `--fuse-conv-bn`: 是否融合 BN 和 Conv 层。该操作会略微提升模型推理速度。
- `EVAL_METRICS`：测试指标。其可选值与对应数据集相关，如 `mAP`，适用于 COCO 等数据集，`PCK` `AUC` `EPE` 适用于 OneHand10K 等数据集等。
- `--gpu-collect`：如果被指定，姿态估计结果将会通过 GPU 通信进行收集。否则，它将被存储到不同 GPU 上的 `TMPDIR` 文件夹中，并在 rank 0 的进程中被收集。
- `TMPDIR`：用于存储不同进程收集的结果文件的临时文件夹。该变量仅当 `--gpu-collect` 没有被指定时有效。
- `CFG_OPTIONS`：覆盖配置文件中的一些实验设置。比如，可以设置'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'，在线修改配置文件内容。
- `JOB_LAUNCHER`：分布式任务初始化启动器选项。可选值有 `none`，`pytorch`，`slurm`，`mpi`。特别地，如果被设置为 `none`, 则会以非分布式模式进行测试。
- `LOCAL_RANK`：本地 rank 的 ID。如果没有被指定，则会被设置为 0。

例子：

假定用户将下载的模型权重文件放置在 `checkpoints/` 目录下。

1. 在 COCO 数据集下测试 ResNet50（不存储测试结果为文件），并验证 `mAP` 指标

   ```shell
   ./tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
       checkpoints/SOME_CHECKPOINT.pth 1 \
       --eval mAP
   ```

1. 使用 8 块 GPU 在 COCO 数据集下测试 ResNet。在线下载模型权重，并验证 `mAP` 指标。

   ```shell
   ./tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
       https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth 8 \
       --eval mAP
   ```

1. 在 slurm 分布式环境中测试 ResNet50 在 COCO 数据集下的 `mAP` 指标

   ```shell
   ./tools/slurm_test.sh slurm_partition test_job \
       configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
       checkpoints/SOME_CHECKPOINT.pth \
       --eval mAP
   ```

### 运行演示

我们提供了丰富的脚本，方便大家快速运行演示。
下面是 多人人体姿态估计 的演示示例，此处我们使用了人工标注的人体框作为输入。

```shell
python demo/top_down_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --json-file ${JSON_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

例子：

```shell
python demo/top_down_img_demo.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results
```

更多实例和细节可以查看 [demo文件夹](/demo) 和 [demo文档](https://mmpose.readthedocs.io/en/latest/demo.html)。

## 如何训练模型

MMPose 使用 `MMDistributedDataParallel` 进行分布式训练，使用 `MMDataParallel` 进行非分布式训练。

对于单机多卡与多台机器的情况，MMPose 使用分布式训练。假设服务器有 8 块 GPU，则会启动 8 个进程，并且每台 GPU 对应一个进程。

每个进程拥有一个独立的模型，以及对应的数据加载器和优化器。
模型参数同步只发生于最开始。之后，每经过一次前向与后向计算，所有 GPU 中梯度都执行一次 allreduce 操作，而后优化器将更新模型参数。
由于梯度执行了 allreduce 操作，因此不同 GPU 中模型参数将保持一致。

### 训练配置

所有的输出（日志文件和模型权重文件）会被将保存到工作目录下。工作目录通过配置文件中的参数 `work_dir` 指定。

默认情况下，MMPose 在每轮训练轮后会在验证集上评估模型，可以通过在训练配置中修改 `interval` 参数来更改评估间隔

```python
evaluation = dict(interval=5)  # 每 5 轮训练进行一次模型评估
```

根据 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)，当 GPU 数量或每个 GPU 上的视频批大小改变时，用户可根据批大小按比例地调整学习率，如，当 4 GPUs x 2 video/gpu 时，lr=0.01；当 16 GPUs x 4 video/gpu 时，lr=0.08。

### 使用单个 GPU 训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如果用户想在命令中指定工作目录，则需要增加参数 `--work-dir ${YOUR_WORK_DIR}`

### 使用 CPU 训练

使用 CPU 训练的流程和使用单 GPU 训练的流程一致，我们仅需要在训练流程开始前禁用 GPU。

```shell
export CUDA_VISIBLE_DEVICES=-1
```

之后运行单 GPU 训练脚本即可。

**注意**：

我们不推荐用户使用 CPU 进行训练，这太过缓慢。我们支持这个功能是为了方便用户在没有 GPU 的机器上进行调试。

### 使用多个 GPU 训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

可选参数为：

- `--work-dir ${WORK_DIR}`：覆盖配置文件中指定的工作目录。
- `--resume-from ${CHECKPOINT_FILE}`：从以前的模型权重文件恢复训练。
- `--no-validate`: 在训练过程中，不进行验证。
- `--gpus ${GPU_NUM}`：使用的 GPU 数量，仅适用于非分布式训练。
- `--gpu-ids ${GPU_IDS}`：使用的 GPU ID，仅适用于非分布式训练。
- `--seed ${SEED}`：设置 python，numpy 和 pytorch 里的种子 ID，已用于生成随机数。
- `--deterministic`：如果被指定，程序将设置 CUDNN 后端的确定化选项。
- `--cfg-options CFG_OPTIONS`:覆盖配置文件中的一些实验设置。比如，可以设置'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'，在线修改配置文件内容。
- `--launcher ${JOB_LAUNCHER}`：分布式任务初始化启动器选项。可选值有 `none`，`pytorch`，`slurm`，`mpi`。特别地，如果被设置为 `none`, 则会以非分布式模式进行测试。
- `--autoscale-lr`:根据 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)，当 GPU 数量或每个 GPU 上的视频批大小改变时，用户可根据批大小按比例地调整学习率。
- `LOCAL_RANK`：本地 rank 的 ID。如果没有被指定，则会被设置为 0。

`resume-from` 和 `load-from` 的区别：
`resume-from` 加载模型参数和优化器状态，并且保留检查点所在的训练轮数，常被用于恢复意外被中断的训练。
`load-from` 只加载模型参数，但训练轮数从 0 开始计数，常被用于微调模型。

这里提供一个使用 8 块 GPU 加载 ResNet50 模型权重文件的例子。

```shell
./tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py 8 --resume_from work_dirs/res50_coco_256x192/latest.pth
```

### 使用多台机器训练

如果用户在 [slurm](https://slurm.schedmd.com/) 集群上运行 MMPose，可使用 `slurm_train.sh` 脚本。（该脚本也支持单台机器上训练）

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} [--work-dir ${WORK_DIR}]
```

这里给出一个在 slurm 集群上的 dev 分区使用 16 块 GPU 训练 ResNet50 的例子。
使用 `GPUS_PER_NODE=8` 参数来指定一个有 8 块 GPUS 的 slurm 集群节点，使用 `CPUS_PER_TASK=2` 来指定每个任务拥有2块cpu。

```shell
GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=2 ./tools/slurm_train.sh Test res50 configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py work_dirs/res50_coco_256x192
```

用户可以查看 [slurm_train.sh](/tools/slurm_train.sh) 文件来检查完整的参数和环境变量。

如果用户的多台机器通过 Ethernet 连接，则可以参考 pytorch [launch utility](https://pytorch.org/docs/en/stable/distributed.html#launch-utility)。如果用户没有高速网络，如 InfiniBand，速度将会非常慢。

### 使用单台机器启动多个任务

如果用使用单台机器启动多个任务，如在有 8 块 GPU 的单台机器上启动 2 个需要 4 块 GPU 的训练任务，则需要为每个任务指定不同端口，以避免通信冲突。

如果用户使用 `dist_train.sh` 脚本启动训练任务，则可以通过以下命令指定端口

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

如果用户在 slurm 集群下启动多个训练任务，则需要修改配置文件（通常是配置文件的第 4 行）中的 `dist_params` 变量，以设置不同的通信端口。

在 `config1.py` 中，

```python
dist_params = dict(backend='nccl', port=29500)
```

在 `config2.py` 中，

```python
dist_params = dict(backend='nccl', port=29501)
```

之后便可启动两个任务，分别对应 `config1.py` 和 `config2.py`。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py [--work-dir ${WORK_DIR}]
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py [--work-dir ${WORK_DIR}]
```

## 进阶教程

目前, MMPose 提供了以下更详细的教程：

- [如何编写配置文件](tutorials/0_config.md)
- [如何微调模型](tutorials/1_finetune.md)
- [如何增加新数据集](tutorials/2_new_dataset.md)
- [如何设计数据处理流程](tutorials/3_data_pipeline.md)
- [如何增加新模块](tutorials/4_new_modules.md)
- [如何导出模型为 onnx 格式](tutorials/5_export_model.md)
- [如何自定义模型运行参数](tutorials/6_customize_runtime.md)
