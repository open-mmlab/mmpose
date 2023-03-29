# 高级训练设置

## 恢复训练

恢复训练是指从之前某次训练保存下来的状态开始继续训练，这里的状态包括模型的权重、优化器和优化器参数调整策略的状态。

### 自动恢复训练

用户可以在训练命令最后加上 `--resume` 恢复训练，程序会自动从 `work_dirs` 中加载最新的权重文件恢复训练。如果 `work_dir` 中有最新的 `checkpoint`（例如该训练在上一次训练时被中断），则会从该 `checkpoint` 恢复训练，否则（例如上一次训练还没来得及保存 `checkpoint` 或者启动了新的训练任务）会重新开始训练。

下面是一个恢复训练的示例:

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py --resume
```

### 指定 Checkpoint 恢复训练

你也可以对 `--resume` 指定 `checkpoint` 路径，MMPose 会自动读取该 `checkpoint` 并从中恢复训练，命令如下：

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py \
    --resume work_dirs/td-hm_res50_8xb64-210e_coco-256x192/latest.pth
```

如果你希望手动在配置文件中指定 `checkpoint` 路径，除了设置 `resume=True`，还需要设置 `load_from` 参数。需要注意的是，如果只设置了 `load_from` 而没有设置 `resume=True`，则只会加载 `checkpoint` 中的权重并重新开始训练，而不是接着之前的状态继续训练。

下面的例子与上面指定 `--resume` 参数的例子等价：

```python
resume = True
load_from = 'work_dirs/td-hm_res50_8xb64-210e_coco-256x192/latest.pth'
# model settings
model = dict(
    ## 内容省略 ##
    )
```

## 自动混合精度（AMP）训练

混合精度训练在不改变模型、不降低模型训练精度的前提下，可以缩短训练时间，降低存储需求，因而能支持更大的 batch size、更大模型和尺寸更大的输入的训练。

如果要开启自动混合精度（AMP）训练，在训练命令最后加上 --amp 即可， 命令如下：

```shell
python tools/train.py ${CONFIG_FILE} --amp
```

具体例子如下：

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py  --amp
```

## 设置随机种子

如果想要在训练时指定随机种子，可以使用以下命令：

```shell
python ./tools/train.py \
    ${CONFIG} \                               # 配置文件路径
    --cfg-options randomness.seed=2023 \      # 设置随机种子为 2023
    [randomness.diff_rank_seed=True] \        # 根据 rank 来设置不同的种子。
    [randomness.deterministic=True]           # 把 cuDNN 后端确定性选项设置为 True
# [] 代表可选参数，实际输入命令行时，不用输入 []
```

randomness 有三个参数可设置，具体含义如下：

- `randomness.seed=2023` ，设置随机种子为 `2023`。

- `randomness.diff_rank_seed=True`，根据 `rank` 来设置不同的种子，`diff_rank_seed` 默认为 `False`。

- `randomness.deterministic=True`，把 `cuDNN` 后端确定性选项设置为 `True`，即把 `torch.backends.cudnn.deterministic` 设为 `True`，把 `torch.backends.cudnn.benchmark` 设为 `False`。`deterministic` 默认为 `False`。更多细节见 [Pytorch Randomness](https://pytorch.org/docs/stable/notes/randomness.html)。

如果你希望手动在配置文件中指定随机种子，可以在配置文件中设置 `random_seed` 参数，具体如下：

```python
randomness = dict(seed=2023)
# model settings
model = dict(
    ## 内容省略 ##
    )
```

## 使用 Tensorboard 可视化训练过程

安装 Tensorboard 环境

```shell
pip install tensorboard
```

在 config 文件中添加 tensorboard 配置

```python
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
```

运行训练命令后，tensorboard 文件会生成在可视化文件夹 `work_dir/${CONFIG}/${TIMESTAMP}/vis_data` 下，运行下面的命令就可以在网页链接使用 tensorboard 查看 loss、学习率和精度等信息。

```shell
tensorboard --logdir work_dir/${CONFIG}/${TIMESTAMP}/vis_data
```
