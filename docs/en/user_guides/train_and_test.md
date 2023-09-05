# Training and Testing

## Launch training

### Train with your PC

You can use `tools/train.py` to train a model on a single machine with a CPU and optionally a GPU.

Here is the full usage of the script:

```shell
python tools/train.py ${CONFIG_FILE} [ARGS]
```

```{note}
By default, MMPose prefers GPU to CPU. If you want to train a model on CPU, please empty `CUDA_VISIBLE_DEVICES` or set it to -1 to make GPU invisible to the program.
```

```shell
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [ARGS]
```

| ARGS                                  | Description                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | The path to the config file.                                                                                                                                        |
| `--work-dir WORK_DIR`                 | The target folder to save logs and checkpoints. Defaults to a folder with the same name as the config file under `./work_dirs`.                                     |
| `--resume [RESUME]`                   | Resume training. If specify a path, resume from it, while if not specify, try to auto resume from the latest checkpoint.                                            |
| `--amp`                               | Enable automatic-mixed-precision training.                                                                                                                          |
| `--no-validate`                       | **Not suggested**. Disable checkpoint evaluation during training.                                                                                                   |
| `--auto-scale-lr`                     | Automatically rescale the learning rate according to the actual batch size and the original batch size.                                                             |
| `--cfg-options CFG_OPTIONS`           | Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either `key="[a,b]"` or `key=a,b`. The argument also allows nested list/tuple values, e.g. `key="[(a,b),(c,d)]"`. Note that quotation marks are necessary and that **no white space is allowed**. |
| `--show-dir SHOW_DIR`                 | The directory to save the result visualization images generated during validation.                                                                                  |
| `--show`                              | Visualize the prediction result in a window.                                                                                                                        |
| `--interval INTERVAL`                 | The interval of samples to visualize.                                                                                                                               |
| `--wait-time WAIT_TIME`               | The display time of every window (in seconds). Defaults to 1.                                                                                                       |
| `--launcher {none,pytorch,slurm,mpi}` | Options for job launcher.                                                                                                                                           |

### Train with multiple GPUs

We provide a shell script to start a multi-GPUs task with `torch.distributed.launch`.

```shell
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]
```

| ARGS          | Description                                                                        |
| ------------- | ---------------------------------------------------------------------------------- |
| `CONFIG_FILE` | The path to the config file.                                                       |
| `GPU_NUM`     | The number of GPUs to be used.                                                     |
| `[PYARGS]`    | The other optional arguments of `tools/train.py`, see [here](#train-with-your-pc). |

You can also specify extra arguments of the launcher by environment variables. For example, change the
communication port of the launcher to 29666 by the below command:

```shell
PORT=29666 bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]
```

If you want to startup multiple training jobs and use different GPUs, you can launch them by specifying
different port and visible devices.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ${CONFIG_FILE1} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=29501 bash ./tools/dist_train.sh ${CONFIG_FILE2} 4 [PY_ARGS]
```

### Train with multiple machines

#### Multiple machines in the same network

If you launch a training job with multiple machines connected with ethernet, you can run the following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

Compared with multi-GPUs in a single machine, you need to specify some extra environment variables:

| ENV_VARS      | Description                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| `NNODES`      | The total number of machines.                                                |
| `NODE_RANK`   | The index of the local machine.                                              |
| `PORT`        | The communication port, it should be the same in all machines.               |
| `MASTER_ADDR` | The IP address of the master machine, it should be the same in all machines. |

Usually, it is slow if you do not have high-speed networking like InfiniBand.

#### Multiple machines managed with slurm

If you run MMPose on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
[ENV_VARS] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [PY_ARGS]
```

Here are the arguments description of the script.

| ARGS          | Description                                                                        |
| ------------- | ---------------------------------------------------------------------------------- |
| `PARTITION`   | The partition to use in your cluster.                                              |
| `JOB_NAME`    | The name of your job, you can name it as you like.                                 |
| `CONFIG_FILE` | The path to the config file.                                                       |
| `WORK_DIR`    | The target folder to save logs and checkpoints.                                    |
| `[PYARGS]`    | The other optional arguments of `tools/train.py`, see [here](#train-with-your-pc). |

Here are the environment variables that can be used to configure the slurm job.

| ENV_VARS        | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `GPUS`          | The total number of GPUs to be used. Defaults to 8.                                                        |
| `GPUS_PER_NODE` | The number of GPUs to be allocated per node. Defaults to 8.                                                |
| `CPUS_PER_TASK` | The number of CPUs to be allocated per task (Usually one GPU corresponds to one task). Defaults to 5.      |
| `SRUN_ARGS`     | The other arguments of `srun`. Available options can be found [here](https://slurm.schedmd.com/srun.html). |

## Resume training

Resume training means to continue training from the state saved from one of the previous trainings, where the state includes the model weights, the state of the optimizer and the optimizer parameter adjustment strategy.

### Automatically resume training

Users can add `--resume` to the end of the training command to resume training. The program will automatically load the latest weight file from `work_dirs` to resume training. If there is a latest `checkpoint` in `work_dirs` (e.g. the training was interrupted during the previous training), the training will be resumed from the `checkpoint`. Otherwise (e.g. the previous training did not save `checkpoint` in time or a new training task was started), the training will be restarted.

Here is an example of resuming training:

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py --resume
```

### Specify the checkpoint to resume training

You can also specify the `checkpoint` path for `--resume`. MMPose will automatically read the `checkpoint` and resume training from it. The command is as follows:

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py \
    --resume work_dirs/td-hm_res50_8xb64-210e_coco-256x192/latest.pth
```

If you hope to manually specify the `checkpoint` path in the config file, in addition to setting `resume=True`, you also need to set the `load_from`.

It should be noted that if only `load_from` is set without setting `resume=True`, only the weights in the `checkpoint` will be loaded and the training will be restarted from scratch, instead of continuing from the previous state.

The following example is equivalent to the example above that specifies the `--resume` parameter:

```python
resume = True
load_from = 'work_dirs/td-hm_res50_8xb64-210e_coco-256x192/latest.pth'
# model settings
model = dict(
    ## omitted ##
    )
```

## Freeze partial parameters during training

In some scenarios, it might be desirable to freeze certain parameters of a model during training to fine-tune specific parts or to prevent overfitting. In MMPose, you can set different hyperparameters for any module in the model by setting custom_keys in `paramwise_cfg`. This allows you to control the learning rate and decay coefficient for specific parts of the model.

For example, if you want to freeze the parameters in `backbone.layer0` and `backbone.layer1`, you can modify the optimizer wrapper in the config file as:

```python
optim_wrapper = dict(
    optimizer=dict(...),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0, decay_mult=0),
            'backbone.layer0': dict(lr_mult=0, decay_mult=0),
        }))
```

This configuration will freeze the parameters in `backbone.layer0` and `backbone.layer1` by setting their learning rate and decay coefficient to 0. By using this approach, you can effectively control the training process and fine-tune specific parts of your model as needed.

## Automatic Mixed Precision (AMP) training

Mixed precision training can reduce training time and storage requirements without changing the model or reducing the model training accuracy, thus supporting larger batch sizes, larger models, and larger input sizes.

To enable Automatic Mixing Precision (AMP) training, add `--amp` to the end of the training command, which is as follows:

```shell
python tools/train.py ${CONFIG_FILE} --amp
```

Specific examples are as follows:

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py  --amp
```

## Set the random seed

If you want to specify the random seed during training, you can use the following command:

```shell
python ./tools/train.py \
    ${CONFIG} \                               # config file
    --cfg-options randomness.seed=2023 \      # set the random seed = 2023
    [randomness.diff_rank_seed=True] \        # Set different seeds according to rank.
    [randomness.deterministic=True]           # Set the cuDNN backend deterministic option to True
# `[]` stands for optional parameters, when actually entering the command line, you do not need to enter `[]`
```

`randomness` has three parameters that can be set, with the following meanings.

- `randomness.seed=2023`, set the random seed to `2023`.

- `randomness.diff_rank_seed=True`, set different seeds according to global `rank`. Defaults to `False`.

- `randomness.deterministic=True`, set the deterministic option for `cuDNN` backend, i.e., set `torch.backends.cudnn.deterministic` to `True` and `torch.backends.cudnn.benchmark` to `False`. Defaults to `False`. See [Pytorch Randomness](https://pytorch.org/docs/stable/notes/randomness.html) for more details.

## Training Log

During training, the training log will be printed in the console as follows:

```shell
07/14 08:26:50 - mmengine - INFO - Epoch(train) [38][ 6/38]  base_lr: 5.148343e-04 lr: 5.148343e-04  eta: 0:15:34  time: 0.540754  data_time: 0.394292  memory: 3141  loss: 0.006220  loss_kpt: 0.006220  acc_pose: 1.000000
```

The training log contains the following information:

- `07/14 08:26:50`: The current time.
- `mmengine`: The name of the program.
- `INFO` or `WARNING`: The log level.
- `Epoch(train)`: The current training stage. `train` means the training stage, `val` means the validation stage.
- `[38][ 6/38]`: The current epoch and the current iteration.
- `base_lr`: The base learning rate.
- `lr`: The current (real) learning rate.
- `eta`: The estimated time of arrival.
- `time`: The elapsed time (minutes) of the current iteration.
- `data_time`: The elapsed time (minutes) of data processing (i/o and transforms).
- `memory`: The GPU memory (MB) allocated by the program.
- `loss`: The total loss value of the current iteration.
- `loss_kpt`: The loss value you passed in head module.
- `acc_pose`: The accuracy value you passed in head module.

## Visualize training process

Monitoring the training process is essential for understanding the performance of your model and making necessary adjustments. In this section, we will introduce two methods to visualize the training process of your MMPose model: TensorBoard and the MMEngine Visualizer.

### TensorBoard

TensorBoard is a powerful tool that allows you to visualize the changes in losses during training. To enable TensorBoard visualization, you may need to:

1. Install TensorBoard environment

   ```shell
   pip install tensorboard
   ```

2. Enable TensorBoard in the config file

   ```python
   visualizer = dict(vis_backends=[
       dict(type='LocalVisBackend'),
       dict(type='TensorboardVisBackend'),
   ])
   ```

The event file generated by TensorBoard will be save under the experiment log folder `${WORK_DIR}`, which defaults to `work_dir/${CONFIG}` or can be specified using the `--work-dir` option. To visualize the training process, use the following command:

```shell
tensorboard --logdir ${WORK_DIR}/${TIMESTAMP}/vis_data
```

### MMEngine visualizer

MMPose also supports visualizing model inference results during validation. To activate this function, please use the `--show` option or set `--show-dir` when launching training. This feature provides an effective way to analyze the model's performance on specific examples and make any necessary adjustments.

## Test your model

### Test with your PC

You can use `tools/test.py` to test a model on a single machine with a CPU and optionally a GPU.

Here is the full usage of the script:

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

```{note}
By default, MMPose prefers GPU to CPU. If you want to test a model on CPU, please empty `CUDA_VISIBLE_DEVICES` or set it to -1 to make GPU invisible to the program.
```

```shell
CUDA_VISIBLE_DEVICES=-1 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

| ARGS                                  | Description                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | The path to the config file.                                                                                                                                        |
| `CHECKPOINT_FILE`                     | The path to the checkpoint file (It can be a http link, and you can find checkpoints [here](https://MMPose.readthedocs.io/en/latest/model_zoo.html)).               |
| `--work-dir WORK_DIR`                 | The directory to save the file containing evaluation metrics.                                                                                                       |
| `--out OUT`                           | The path to save the file containing evaluation metrics.                                                                                                            |
| `--dump DUMP`                         | The path to dump all outputs of the model for offline evaluation.                                                                                                   |
| `--cfg-options CFG_OPTIONS`           | Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either `key="[a,b]"` or `key=a,b`. The argument also allows nested list/tuple values, e.g. `key="[(a,b),(c,d)]"`. Note that quotation marks are necessary and that no white space is allowed. |
| `--show-dir SHOW_DIR`                 | The directory to save the result visualization images.                                                                                                              |
| `--show`                              | Visualize the prediction result in a window.                                                                                                                        |
| `--interval INTERVAL`                 | The interval of samples to visualize.                                                                                                                               |
| `--wait-time WAIT_TIME`               | The display time of every window (in seconds). Defaults to 1.                                                                                                       |
| `--launcher {none,pytorch,slurm,mpi}` | Options for job launcher.                                                                                                                                           |

### Test with multiple GPUs

We provide a shell script to start a multi-GPUs task with `torch.distributed.launch`.

```shell
bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

| ARGS              | Description                                                                                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`     | The path to the config file.                                                                                                                          |
| `CHECKPOINT_FILE` | The path to the checkpoint file (It can be a http link, and you can find checkpoints [here](https://mmpose.readthedocs.io/en/latest/model_zoo.html)). |
| `GPU_NUM`         | The number of GPUs to be used.                                                                                                                        |
| `[PYARGS]`        | The other optional arguments of `tools/test.py`, see [here](#test-with-your-pc).                                                                      |

You can also specify extra arguments of the launcher by environment variables. For example, change the
communication port of the launcher to 29666 by the below command:

```shell
PORT=29666 bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

If you want to startup multiple test jobs and use different GPUs, you can launch them by specifying
different port and visible devices.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_test.sh ${CONFIG_FILE1} ${CHECKPOINT_FILE} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash ./tools/dist_test.sh ${CONFIG_FILE2} ${CHECKPOINT_FILE} 4 [PY_ARGS]
```

### Test with multiple machines

#### Multiple machines in the same network

If you launch a test job with multiple machines connected with ethernet, you can run the following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT_FILE $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT_FILE $GPUS
```

Compared with multi-GPUs in a single machine, you need to specify some extra environment variables:

| ENV_VARS      | Description                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| `NNODES`      | The total number of machines.                                                |
| `NODE_RANK`   | The index of the local machine.                                              |
| `PORT`        | The communication port, it should be the same in all machines.               |
| `MASTER_ADDR` | The IP address of the master machine, it should be the same in all machines. |

Usually, it is slow if you do not have high-speed networking like InfiniBand.

#### Multiple machines managed with slurm

If you run MMPose on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`.

```shell
[ENV_VARS] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE} [PY_ARGS]
```

Here are the argument descriptions of the script.

| ARGS              | Description                                                                                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PARTITION`       | The partition to use in your cluster.                                                                                                                 |
| `JOB_NAME`        | The name of your job, you can name it as you like.                                                                                                    |
| `CONFIG_FILE`     | The path to the config file.                                                                                                                          |
| `CHECKPOINT_FILE` | The path to the checkpoint file (It can be a http link, and you can find checkpoints [here](https://MMPose.readthedocs.io/en/latest/model_zoo.html)). |
| `[PYARGS]`        | The other optional arguments of `tools/test.py`, see [here](#test-with-your-pc).                                                                      |

Here are the environment variables that can be used to configure the slurm job.

| ENV_VARS        | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `GPUS`          | The total number of GPUs to be used. Defaults to 8.                                                        |
| `GPUS_PER_NODE` | The number of GPUs to be allocated per node. Defaults to 8.                                                |
| `CPUS_PER_TASK` | The number of CPUs to be allocated per task (Usually one GPU corresponds to one task). Defaults to 5.      |
| `SRUN_ARGS`     | The other arguments of `srun`. Available options can be found [here](https://slurm.schedmd.com/srun.html). |

## Custom Testing Features

### Test with Custom Metrics

If you're looking to assess models using unique metrics not already supported by MMPose, you'll need to code these metrics yourself and include them in your config file. For guidance on how to accomplish this, check out our [customized evaluation guide](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_evaluation.html).

### Evaluating Across Multiple Datasets

MMPose offers a handy tool known as `MultiDatasetEvaluator` for streamlined assessment across multiple datasets. Setting up this evaluator in your config file is a breeze. Below is a quick example demonstrating how to evaluate a model using both the COCO and AIC datasets:

```python
# Set up validation datasets
coco_val = dict(type='CocoDataset', ...)
aic_val = dict(type='AicDataset', ...)
val_dataset = dict(
        type='CombinedDataset',
        datasets=[coco_val, aic_val],
        pipeline=val_pipeline,
        ...)

# configurate the evaluator
val_evaluator = dict(
    type='MultiDatasetEvaluator',
    metrics=[  # metrics for each dataset
        dict(type='CocoMetric',
             ann_file='data/coco/annotations/person_keypoints_val2017.json'),
        dict(type='CocoMetric',
            ann_file='data/aic/annotations/aic_val.json',
            use_area=False,
            prefix='aic')
    ],
    # the number and order of datasets must align with metrics
    datasets=[coco_val, aic_val],
    )
```

Keep in mind that different datasets, like COCO and AIC, have various keypoint definitions. Yet, the model's output keypoints are standardized. This results in a discrepancy between the model outputs and the actual ground truth. To address this, you can employ `KeypointConverter` to align the keypoint configurations between different datasets. Here’s a full example that shows how to leverage `KeypointConverter` to align AIC keypoints with COCO keypoints:

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

For further clarification on converting AIC keypoints to COCO keypoints, please consult [this guide](https://mmpose.readthedocs.io/en/latest/user_guides/mixed_datasets.html#merge-aic-into-coco).
