# 恢复训练

恢复训练是指从之前某次训练保存下来的状态开始继续训练，这里的状态包括模型的权重、优化器和优化器参数调整策略的状态。

## 自动恢复训练

用户可以在训练命令最后加上 `--resume` 恢复训练，程序会自动从 `work_dirs` 中加载最新的权重文件恢复训练。如果 `work_dir` 中有最新的 `checkpoint`（例如该训练在上一次训练时被中断），则会从该 `checkpoint` 恢复训练，否则（例如上一次训练还没来得及保存 `checkpoint` 或者启动了新的训练任务）会重新开始训练。

下面是一个恢复训练的示例:

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py --resume
```

## 指定 Checkpoint 恢复训练

你也可以对 `--resume` 指定 `checkpoint` 路径，MMPose 会自动读取该 `checkpoint` 并从中恢复训练，命令如下：

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py --resume work_dirs/td-hm_res50_8xb64-210e_coco-256x192/latest.pth
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
