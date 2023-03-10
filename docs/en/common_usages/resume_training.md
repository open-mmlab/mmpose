# Resume Training

Resume training means to continue training from the state saved from one of the previous trainings, where the state includes the model weights, the state of the optimizer and the optimizer parameter adjustment strategy.

## Automatically resume training

Users can add `--resume` to the end of the training command to resume training. The program will automatically load the latest weight file from `work_dirs` to resume training. If there is a latest `checkpoint` in `work_dirs` (e.g. the training was interrupted during the previous training), the training will be resumed from the `checkpoint`. Otherwise (e.g. the previous training did not save `checkpoint` in time or a new training task was started), the training will be restarted.

Here is an example of resuming training:

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py --resume
```

## Specify the checkpoint to resume training

You can also specify the `checkpoint` path for `--resume`. MMPose will automatically read the `checkpoint` and resume training from it. The command is as follows:

```shell
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py --resume work_dirs/td-hm_res50_8xb64-210e_coco-256x192/latest.pth
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
