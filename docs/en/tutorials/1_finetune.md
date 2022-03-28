# Tutorial 1: Finetuning Models

Detectors pre-trained on the COCO dataset can serve as a good pre-trained model for other datasets, e.g., COCO-WholeBody Dataset.
This tutorial provides instruction for users to use the models provided in the [Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html) for other datasets to obtain better performance.

<!-- TOC -->

- [Outline](#outline)
- [Modify Head](#modify-head)
- [Modify Dataset](#modify-dataset)
- [Modify Training Schedule](#modify-training-schedule)
- [Use Pre-Trained Model](#use-pre-trained-model)

<!-- TOC -->

## Outline

There are two steps to finetune a model on a new dataset.

- Add support for the new dataset following [Tutorial 2: Adding New Dataset](tutorials/../2_new_dataset.md).
- Modify the configs as will be discussed in this tutorial.

To finetune on the custom datasets, the users need to modify four parts in the config.

## Modify Head

Then the new config needs to modify the model according to the keypoint numbers of the new datasets. By only changing `out_channels` in the keypoint_head.
For example, we have 133 keypoints for COCO-WholeBody, and we have 17 keypoints for COCO.

```python
channel_cfg = dict(
    num_output_channels=133,  # changing from 17 to 133
    dataset_joints=133,  # changing from 17 to 133
    dataset_channel=[
        list(range(133)),  # changing from 17 to 133
    ],
    inference_channel=list(range(133)))  # changing from 17 to 133

# model settings
model = dict(
    type='TopDown',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w48-8ef0771d.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=48,
        out_channels=channel_cfg['num_output_channels'], # modify this
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='unbiased',
        shift_heatmap=True,
        modulate_kernel=17))
```

Note that the `pretrained='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'` setting is used for initializing backbone.
If you are training a new model from ImageNet-pretrained weights, this is for you.
However, this setting is not related to our task at hand. What we need is load_from, which will be discussed later.

## Modify dataset

The users may also need to prepare the dataset and write the configs about dataset.
MMPose supports multiple (10+) dataset, including COCO, COCO-WholeBody and MPII-TRB.

```python
data_root = 'data/coco'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCocoWholeBodyDataset', # modify the name of the dataset
        ann_file=f'{data_root}/annotations/coco_wholebody_train_v1.0.json', # modify the path to the annotation file
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownCocoWholeBodyDataset', # modify the name of the dataset
        ann_file=f'{data_root}/annotations/coco_wholebody_val_v1.0.json', # modify the path to the annotation file
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownCocoWholeBodyDataset', # modify the name of the dataset
        ann_file=f'{data_root}/annotations/coco_wholebody_val_v1.0.json', # modify the path to the annotation file
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline)
)
```

## Modify training schedule

The finetuning hyperparameters vary from the default schedule. It usually requires smaller learning rate and less training epochs

```python
optimizer = dict(
    type='Adam',
    lr=5e-4, # reduce it
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200]) # reduce it
total_epochs = 210 # reduce it
```

## Use pre-trained model

Users can load a pre-trained model by setting the `load_from` field of the config to the model's path or link.
The users might need to download the model weights before training to avoid the download time during training.

```python
# use the pre-trained model for the whole HRNet
load_from = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth'  # model path can be found in model zoo
```
