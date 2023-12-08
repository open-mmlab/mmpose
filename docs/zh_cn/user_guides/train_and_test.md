# 训练与测试

## 启动训练

### 本地训练

你可以使用 `tools/train.py` 在单机上使用 CPU 或单个 GPU 训练模型。

```shell
python tools/train.py ${CONFIG_FILE} [ARGS]
```

```{note}
默认情况下，MMPose 会优先使用 GPU 而不是 CPU。如果你想在 CPU 上训练模型，请清空 `CUDA_VISIBLE_DEVICES` 或将其设置为 -1，使 GPU 对程序不可见。
```

```shell
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [ARGS]
```

| ARGS                                  | Description                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | 配置文件路径                                                                                                                                                        |
| `--work-dir WORK_DIR`                 | 训练日志与 checkpoint 存放目录，默认使用配置文件名作为目录存放在 `./work_dirs` 下                                                                                   |
| `--resume [RESUME]`                   | 恢复训练，可以从指定 checkpoint 进行重启，不指定则会使用最近一次的 checkpoint                                                                                       |
| `--amp`                               | 开启混合精度训练                                                                                                                                                    |
| `--no-validate`                       | **不建议新手开启**。 训练中不进行评测                                                                                                                               |
| `--auto-scale-lr`                     | 自动根据当前设置的实际 batch size 和配置文件中的标准 batch size 进行学习率缩放                                                                                      |
| `--cfg-options CFG_OPTIONS`           | 对当前配置文件中的一些设置进行临时覆盖，字典 key-value 格式为 xxx=yyy。如果需要覆盖的值是一个数组，格式应当为 `key="[a,b]"` 或 `key=a,b`。也允许使用元组，如 `key="[(a,b),(c,d)]"`。注意双引号是**必须的**，且**不允许**使用空格。 |
| `--show-dir SHOW_DIR`                 | 验证阶段生成的可视化图片存放路径                                                                                                                                    |
| `--show`                              | 使用窗口显示预测的可视化结果                                                                                                                                        |
| `--interval INTERVAL`                 | 进行可视化的间隔（每隔多少张图可视化一张）                                                                                                                          |
| `--wait-time WAIT_TIME`               | 可视化显示时每张图片的持续时间（单位：秒），默认为 1                                                                                                                |
| `--launcher {none,pytorch,slurm,mpi}` | 可选的启动器                                                                                                                                                        |

### 多卡训练

我们提供了一个脚本来使用 `torch.distributed.launch` 启动多卡训练。

```shell
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]
```

| ARGS          | Description                                        |
| ------------- | -------------------------------------------------- |
| `CONFIG_FILE` | 配置文件路径                                       |
| `GPU_NUM`     | 使用 GPU 数量                                      |
| `[PYARGS]`    | 其他配置项 `tools/train.py`, 见 [这里](#本地训练). |

你也可以通过环境变量来指定启动器的额外参数。例如，通过以下命令将启动器的通信端口改为 29666：

```shell
PORT=29666 bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]
```

如果你想同时启动多个训练任务并使用不同的 GPU，你可以通过指定不同的端口和可见设备来启动它们。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ${CONFIG_FILE1} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=29501 bash ./tools/dist_train.sh ${CONFIG_FILE2} 4 [PY_ARGS]
```

### 分布式训练

#### 局域网多机训练

如果你使用以太网连接的多台机器启动训练任务，你可以运行以下命令：

在第一台机器上：

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

相比于单机多卡，你需要指定一些额外的环境变量：

| 环境变量      | 描述                       |
| ------------- | -------------------------- |
| `NNODES`      | 机器总数                   |
| `NODE_RANK`   | 当前机器序号               |
| `PORT`        | 通信端口，所有机器必须相同 |
| `MASTER_ADDR` | 主机地址，所有机器必须相同 |

通常情况下，如果你没有像 InfiniBand 这样的高速网络，那么训练速度会很慢。

#### Slurm 多机训练

如果你在一个使用 [slurm](https://slurm.schedmd.com/) 管理的集群上运行 MMPose，你可以使用 `slurm_train.sh` 脚本。

```shell
[ENV_VARS] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [PY_ARGS]
```

脚本参数说明：

| 参数          | 描述                                               |
| ------------- | -------------------------------------------------- |
| `PARTITION`   | 指定集群分区                                       |
| `JOB_NAME`    | 任务名，可以任取                                   |
| `CONFIG_FILE` | 配置文件路径                                       |
| `WORK_DIR`    | 训练日志存储路径                                   |
| `[PYARGS]`    | 其他配置项 `tools/train.py`, 见 [这里](#本地训练). |

以下是可以用来配置 slurm 任务的环境变量：

| 环境变量        | 描述                                                                     |
| --------------- | ------------------------------------------------------------------------ |
| `GPUS`          | GPU 总数，默认为 8                                                       |
| `GPUS_PER_NODE` | 每台机器使用的 GPU 总数，默认为 8                                        |
| `CPUS_PER_TASK` | 每个任务分配的 CPU 总数（通常为 1 张 GPU 对应 1 个任务进程），默认为 5   |
| `SRUN_ARGS`     | `srun` 的其他参数，可选项见 [这里](https://slurm.schedmd.com/srun.html). |

## 恢复训练

恢复训练意味着从之前的训练中保存的状态继续训练，其中状态包括模型权重、优化器状态和优化器参数调整策略的状态。

### 自动恢复

用户可以在训练命令的末尾添加 `--resume` 来恢复训练。程序会自动从 `work_dirs` 中加载最新的权重文件来恢复训练。如果 `work_dirs` 中有最新的 `checkpoint`（例如在之前的训练中中断了训练），则会从 `checkpoint` 处恢复训练。否则（例如之前的训练没有及时保存 `checkpoint` 或者启动了一个新的训练任务），则会重新开始训练。

以下是一个恢复训练的例子：

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py --resume
```

### 指定 checkpoint 恢复

你可以在 `load_from` 中指定 `checkpoint` 的路径，MMPose 会自动读取 `checkpoint` 并从中恢复训练。命令如下：

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py \
    --resume work_dirs/td-hm_res50_8xb64-210e_coco-256x192/latest.pth
```

如果你希望在配置文件中手动指定 `checkpoint` 路径，除了设置 `resume=True`，还需要设置 `load_from`。

需要注意的是，如果只设置了 `load_from` 而没有设置 `resume=True`，那么只会加载 `checkpoint` 中的权重，而不会从之前的状态继续训练。

以下的例子与上面指定 `--resume` 参数的例子等价：

```Python
resume = True
load_from = 'work_dirs/td-hm_res50_8xb64-210e_coco-256x192/latest.pth'
# model settings
model = dict(
    ## omitted ##
    )
```

## 在训练中冻结部分参数

在某些场景下，我们可能希望在训练过程中冻结模型的某些参数，以便微调特定部分或防止过拟合。在 MMPose 中，你可以通过在 `paramwise_cfg` 中设置 `custom_keys` 来为模型中的任何模块设置不同的超参数。这样可以让你控制模型特定部分的学习率和衰减系数。

例如，如果你想冻结 `backbone.layer0` 和 `backbone.layer1` 的所有参数，你可以在配置文件中添加以下内容：

```Python
optim_wrapper = dict(
    optimizer=dict(...),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0, decay_mult=0),
            'backbone.layer0': dict(lr_mult=0, decay_mult=0),
        }))
```

以上配置将会通过将学习率和衰减系数设置为 0 来冻结 `backbone.layer0` 和 `backbone.layer1` 中的参数。通过这种方式，你可以有效地控制训练过程，并根据需要微调模型的特定部分。

## 自动混合精度训练（AMP）

混合精度训练可以减少训练时间和存储需求，而不改变模型或降低模型训练精度，从而支持更大的 batch size、更大的模型和更大的输入尺寸。

要启用自动混合精度（AMP）训练，请在训练命令的末尾添加 `--amp`，如下所示：

```shell
python tools/train.py ${CONFIG_FILE} --amp
```

具体例子如下：

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py  --amp
```

## 设置随机种子

如果你想指定随机种子，你可以通过如下命令：

```shell
python ./tools/train.py \
    ${CONFIG} \                               # 配置文件
    --cfg-options randomness.seed=2023 \      # 设置 random seed = 2023
    [randomness.diff_rank_seed=True] \        # 不同 rank 的进程使用不同的随机种子
    [randomness.deterministic=True]           # 设置 cudnn.deterministic=True
# `[]` 表示可选的参数，你不需要输入 `[]`
```

`randomness` 还有三个参数可以设置，具体含义如下。

- `randomness.seed=2023`，将随机种子设置为 `2023`。

- `randomness.diff_rank_seed=True`，根据全局 `rank` 设置不同的随机种子。默认为 `False`。

- `randomness.deterministic=True`，设置 `cuDNN` 后端的确定性选项，即将 `torch.backends.cudnn.deterministic` 设置为 `True`，将 `torch.backends.cudnn.benchmark` 设置为 `False`。默认为 `False`。更多细节请参考 [Pytorch Randomness](https://pytorch.org/docs/stable/notes/randomness.html)。

## 训练日志说明

在训练中，命令行会实时打印训练日志如下：

```shell
07/14 08:26:50 - mmengine - INFO - Epoch(train) [38][ 6/38]  base_lr: 5.148343e-04 lr: 5.148343e-04  eta: 0:15:34  time: 0.540754  data_time: 0.394292  memory: 3141  loss: 0.006220  loss_kpt: 0.006220  acc_pose: 1.000000
```

以上训练日志包括如下内容：

- `07/14 08:26:50`：当前时间
- `mmengine`：日志前缀，表示日志来自 MMEngine
- `INFO` or `WARNING`：日志级别，表示该日志为普通信息
- `Epoch(train)`：当前处于训练阶段，如果处于验证阶段，则为 `Epoch(val)`
- `[38][ 6/38]`：当前处于第 38 个 epoch，当前 batch 为第 6 个 batch，总共有 38 个 batch
- `base_lr`：基础学习率
- `lr`：当前实际使用的学习率
- `eta`：预计训练剩余时间
- `time`：当前 batch 的训练时间（单位：分钟）
- `data_time`：当前 batch 的数据加载（i/o，数据增强）时间（单位：分钟）
- `memory`：当前进程占用的显存（单位：MB）
- `loss`：当前 batch 的总 loss
- `loss_kpt`：当前 batch 的关键点 loss
- `acc_pose`：当前 batch 的姿态准确率

## 可视化训练进程

监视训练过程对于了解模型的性能并进行必要的调整至关重要。在本节中，我们将介绍两种可视化训练过程的方法：TensorBoard 和 MMEngine Visualizer。

### TensorBoard

TensorBoard 是一个强大的工具，可以让你可视化训练过程中的 loss 变化。要启用 TensorBoard 可视化，你可能需要：

1. 安装 TensorBoard

   ```shell
   pip install tensorboard
   ```

2. 在配置文件中开启 TensorBoard 作为可视化后端：

   ```python
   visualizer = dict(vis_backends=[
       dict(type='LocalVisBackend'),
       dict(type='TensorboardVisBackend'),
   ])
   ```

Tensorboard 生成的 event 文件会保存在实验日志文件夹 `${WORK_DIR}` 下，该文件夹默认为 `work_dir/${CONFIG}`，你也可以通过 `--work-dir` 参数指定。要可视化训练过程，请使用以下命令：

```shell
tensorboard --logdir ${WORK_DIR}/${TIMESTAMP}/vis_data
```

### MMEngine Visualizer

MMPose 还支持在验证过程中可视化模型的推理结果。要启用此功能，请在启动训练时使用 `--show` 选项或设置 `--show-dir`。这个功能提供了一种有效的方法来分析模型在特定示例上的性能并进行必要的调整。

## 测试

### 本地测试

你可以使用 `tools/test.py` 在单机上使用 CPU 或单个 GPU 测试模型。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

```{note}
默认情况下，MMPose 会优先使用 GPU 而不是 CPU。如果你想在 CPU 上测试模型，请清空 `CUDA_VISIBLE_DEVICES` 或将其设置为 -1，使 GPU 对程序不可见。
```

```shell
CUDA_VISIBLE_DEVICES=-1 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

| ARGS                                  | Description                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | 配置文件路径file.                                                                                                                                                   |
| `CHECKPOINT_FILE`                     | checkpoint 文件路径，可以是本地文件，也可以是网络链接。 [这里](https://MMPose.readthedocs.io/en/latest/model_zoo.html) 是 MMPose 提供的 checkpoint 列表.            |
| `--work-dir WORK_DIR`                 | 评测结果存储目录                                                                                                                                                    |
| `--out OUT`                           | 评测结果存放文件                                                                                                                                                    |
| `--dump DUMP`                         | 导出评测时的模型输出，用于用户自行离线评测                                                                                                                          |
| `--cfg-options CFG_OPTIONS`           | 对当前配置文件中的一些设置进行临时覆盖，字典 key-value 格式为 xxx=yyy。如果需要覆盖的值是一个数组，格式应当为 `key="[a,b]"` 或 `key=a,b`。也允许使用元组，如 `key="[(a,b),(c,d)]"`。注意双引号是**必须的**，且**不允许**使用空格。 |
| `--show-dir SHOW_DIR`                 | T验证阶段生成的可视化图片存放路径                                                                                                                                   |
| `--show`                              | 使用窗口显示预测的可视化结果                                                                                                                                        |
| `--interval INTERVAL`                 | 进行可视化的间隔（每隔多少张图可视化一张）                                                                                                                          |
| `--wait-time WAIT_TIME`               | 可视化显示时每张图片的持续时间（单位：秒），默认为 1                                                                                                                |
| `--launcher {none,pytorch,slurm,mpi}` | 可选的启动器                                                                                                                                                        |

### 多卡测试

我们提供了一个脚本来使用 `torch.distributed.launch` 启动多卡测试。

```shell
bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

| ARGS              | Description                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`     | 配置文件路径                                                                                                                                            |
| `CHECKPOINT_FILE` | checkpoint 文件路径，可以是本地文件，也可以是网络链接。 [这里](https://MMPose.readthedocs.io/en/latest/model_zoo.html) 是 MMPose 提供的 checkpoint 列表 |
| `GPU_NUM`         | 使用 GPU 数量                                                                                                                                           |
| `[PYARGS]`        | 其他配置项 `tools/test.py`, 见 [这里](#本地测试)                                                                                                        |

你也可以通过环境变量来指定启动器的额外参数。例如，通过以下命令将启动器的通信端口改为 29666：

```shell
PORT=29666 bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

如果你想同时启动多个测试任务并使用不同的 GPU，你可以通过指定不同的端口和可见设备来启动它们。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_test.sh ${CONFIG_FILE1} ${CHECKPOINT_FILE} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash ./tools/dist_test.sh ${CONFIG_FILE2} ${CHECKPOINT_FILE} 4 [PY_ARGS]
```

### 分布式测试

#### 局域网多机测试

如果你使用以太网连接的多台机器启动测试任务，你可以运行以下命令：

在第一台机器上：

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT_FILE $GPUS
```

在第二台机器上：

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT_FILE $GPUS
```

相比于单机多卡，你需要指定一些额外的环境变量：

| 环境变量      | 描述                       |
| ------------- | -------------------------- |
| `NNODES`      | 机器总数                   |
| `NODE_RANK`   | 当前机器序号               |
| `PORT`        | 通信端口，所有机器必须相同 |
| `MASTER_ADDR` | 主机地址，所有机器必须相同 |

通常情况下，如果你没有像 InfiniBand 这样的高速网络，那么测试速度会很慢。

#### Slurm 多机测试

如果你在一个使用 [slurm](https://slurm.schedmd.com/) 管理的集群上运行 MMPose，你可以使用 `slurm_test.sh` 脚本。

```shell
[ENV_VARS] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE} [PY_ARGS]
```

脚本参数说明：

| 参数              | 描述                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PARTITION`       | 指定集群分区                                                                                                                                            |
| `JOB_NAME`        | 任务名，可以任取                                                                                                                                        |
| `CONFIG_FILE`     | 配置文件路径                                                                                                                                            |
| `CHECKPOINT_FILE` | checkpoint 文件路径，可以是本地文件，也可以是网络链接。 [这里](https://MMPose.readthedocs.io/en/latest/model_zoo.html) 是 MMPose 提供的 checkpoint 列表 |
| `[PYARGS]`        | 其他配置项 `tools/test.py`, 见 [这里](#本地测试)                                                                                                        |

以下是可以用来配置 slurm 任务的环境变量：

| 环境变量        | 描述                                                                     |
| --------------- | ------------------------------------------------------------------------ |
| `GPUS`          | GPU 总数，默认为 8                                                       |
| `GPUS_PER_NODE` | 每台机器使用的 GPU 总数，默认为 8                                        |
| `CPUS_PER_TASK` | 每个任务分配的 CPU 总数（通常为 1 张 GPU 对应 1 个任务进程），默认为 5   |
| `SRUN_ARGS`     | `srun` 的其他参数，可选项见 [这里](https://slurm.schedmd.com/srun.html). |

## 自定义测试

### 用自定义度量进行测试

如果您希望使用 MMPose 中尚未支持的独特度量来评估模型，您将需要自己编写这些度量并将它们包含在您的配置文件中。关于如何实现这一点的指导，请查看我们的 [自定义评估指南](https://mmpose.readthedocs.io/zh_CN/dev-1.x/advanced_guides/customize_evaluation.html)。

### 在多个数据集上进行评估

MMPose 提供了一个名为 `MultiDatasetEvaluator` 的便捷工具，用于在多个数据集上进行简化评估。在配置文件中设置此评估器非常简单。下面是一个快速示例，演示如何使用 COCO 和 AIC 数据集评估模型：

```python
# 设置验证数据集
coco_val = dict(type='CocoDataset', ...)

aic_val = dict(type='AicDataset', ...)

val_dataset = dict(
        type='CombinedDataset',
        datasets=[coco_val, aic_val],
        pipeline=val_pipeline,
        ...)

# 配置评估器
val_evaluator = dict(
    type='MultiDatasetEvaluator',
    metrics=[  # 为每个数据集配置度量
        dict(type='CocoMetric',
             ann_file='data/coco/annotations/person_keypoints_val2017.json'),
        dict(type='CocoMetric',
            ann_file='data/aic/annotations/aic_val.json',
            use_area=False,
            prefix='aic')
    ],
    # 数据集个数和顺序与度量必须匹配
    datasets=[coco_val, aic_val],
    )
```

同的数据集（如 COCO 和 AIC）具有不同的关键点定义。然而，模型的输出关键点是标准化的。这导致了模型输出与真值之间关键点顺序的差异。为解决这一问题，您可以使用 `KeypointConverter` 来对齐不同数据集之间的关键点顺序。下面是一个完整示例，展示了如何利用 `KeypointConverter` 来对齐 AIC 关键点与 COCO 关键点：

```python
aic_to_coco_converter = dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[
                (0, 6),
                (1, 8),
                (2, 10),
                (3, 5),
                (4, 7),
                (5, 9),
                (6, 12),
                (7, 14),
                (8, 16),
                (9, 11),
                (10, 13),
                (11, 15),
            ])

# val datasets
coco_val = dict(
    type='CocoDataset',
    data_root='data/coco/',
    data_mode='topdown',
    ann_file='annotations/person_keypoints_val2017.json',
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=[],
)

aic_val = dict(
        type='AicDataset',
        data_root='data/aic/',
        data_mode=data_mode,
        ann_file='annotations/aic_val.json',
        data_prefix=dict(img='ai_challenger_keypoint_validation_20170911/'
                         'keypoint_validation_images_20170911/'),
        test_mode=True,
        pipeline=[],
    )

val_dataset = dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[coco_val, aic_val],
        pipeline=val_pipeline,
        test_mode=True,
    )

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=val_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='MultiDatasetEvaluator',
    metrics=[
        dict(type='CocoMetric',
             ann_file=data_root + 'annotations/person_keypoints_val2017.json'),
        dict(type='CocoMetric',
            ann_file='data/aic/annotations/aic_val.json',
            use_area=False,
            gt_converter=aic_to_coco_converter,
            prefix='aic')
    ],
    datasets=val_dataset['datasets'],
    )

test_evaluator = val_evaluator
```

如需进一步了解如何将 AIC 关键点转换为 COCO 关键点，请查阅 [该指南](https://mmpose.readthedocs.io/zh_CN/dev-1.x/user_guides/mixed_datasets.html#aic-coco)。

### 使用自定义检测器评估 Top-down 模型

要评估 Top-down 模型，您可以使用人工标注的或预先检测到的边界框。 `bbox_file` 提供了由特定检测器生成的这些框。例如，`COCO_val2017_detections_AP_H_56_person.json` 包含了使用具有 56.4 人类 AP 的检测器捕获的 COCO val2017 数据集的边界框。要使用 MMDetection 支持的自定义检测器创建您自己的 `bbox_file`，请运行以下命令：

```sh
python tools/misc/generate_bbox_file.py \
    ${DET_CONFIG} ${DET_WEIGHT} ${OUTPUT_FILE_NAME} \
    [--pose-config ${POSE_CONFIG}] \
    [--score-thr ${SCORE_THRESHOLD}] [--nms-thr ${NMS_THRESHOLD}]
```

其中，`DET_CONFIG` 和 `DET_WEIGHT` 用于创建目标检测器。 `POSE_CONFIG` 指定需要边界框检测的测试数据集。`SCORE_THRESHOLD` 和 `NMS_THRESHOLD` 用于边界框过滤。
