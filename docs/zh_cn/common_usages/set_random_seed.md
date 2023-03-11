# 设置随机种子

如果想要在训练时指定随机种子，可以使用以下命令：

```
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
