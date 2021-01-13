# Tutorial 1: Finetuning Models

Detectors pre-trained on the COCO dataset can serve as a good pre-trained model for other datasets, e.g., PoseTrack Dataset.
This tutorial provides instruction for users to use the models provided in the [Model Zoo](../top_down_models.md) for other datasets to obatin better performance.

There are two steps to finetune a model on a new dataset.

- Add support for the new dataset following [Tutorial 2: Adding New Dataset](tutorials/../2_new_dataset.md).
- Modify the configs as will be discussed in this tutorial.

To finetune on the custom datasets, the users need to modify four parts in the config.

## Modify Model

Then the new config needs to modify the model according to the keypoint numbers of the new datasets. By only changing `out_channels` in the keypoint_head.

```python
# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(type='ResNet', depth=18),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=512,
        out_channels=17,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='unbiased',
        shift_heatmap=True,
        modulate_kernel=11),
    loss_keypoint=dict(type='JointsMSELoss', use_target_weight=False))
```

## Modify dataset

The users may also need to prepare the dataset and write the configs about dataset. MMPose already support COCO and MPII-TRB Dataset.

## Modify training schedule

The finetuning hyperparameters vary from the default schedule. It usually requires smaller learning rate and less training epochs

```python
# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 15])
total_epochs = 20
```

## Use pre-trained model

To use the pre-trained model, the new config add the link of pre-trained models in the `load_from`. The users might need to download the model weights before training to avoid the download time during training.

```python
# use the pre-trained model for the whole Simple Baseline res50-backbone network
load_from = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth'  # model path can be found in model zoo
```
