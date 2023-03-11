# Set the random seed

If you want to specify the random seed during training, you can use the following command:

```shell
python ./tools/train.py \
    ${CONFIG} \                               # config file
    --cfg-options randomness.seed=2023 \      # set the random seed = 2023
    [randomness.diff_rank_seed=True] \        # Set different seeds according to rank.
    [randomness.deterministic=True]           # Set the cuDNN backend deterministic option to True
# `[]` stands for optional parameters, when actually entering the command line, you do not need to enter `[]`
```
