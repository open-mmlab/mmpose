# 教程 1：如何微调模型

在 COCO 数据集上进行预训练，然后在其他数据集（如 COCO-WholeBody 数据集）上进行微调，往往可以提升模型的效果。
本教程介绍如何使用[模型库](https://mmpose.readthedocs.io/en/latest/modelzoo.html)中的预训练模型，并在其他数据集上进行微调。

<!-- TOC -->

- [概要](#概要)
- [修改 Head](#修改网络头)
- [修改数据集](#修改数据集)
- [修改训练策略](#修改训练策略)
- [使用预训练模型](#使用预训练模型)

<!-- TOC -->

## 概要

对新数据集上的模型微调需要两个步骤：

1. 支持新数据集。详情参见 [教程 2：如何增加新数据集](2_new_dataset.md)
2. 修改配置文件。这部分将在本教程中做具体讨论。

例如，如果想要在自定义数据集上，微调 COCO 预训练的模型，则需要修改 [配置文件](0_config.md) 中 网络头、数据集、训练策略、预训练模型四个部分。

## 修改网络头

如果自定义数据集的关键点个数，与 COCO 不同，则需要相应修改 `keypoint_head` 中的 `out_channels` 参数。
网络头（head）的最后一层的预训练参数不会被载入，而其他层的参数都会被正常载入。
例如，COCO-WholeBody 拥有 133 个关键点，因此需要把 17 （COCO 数据集的关键点数目） 改为 133。

```python
channel_cfg = dict(
    num_output_channels=133,  # 从 17 改为 133
    dataset_joints=133,  # 从 17 改为 133
    dataset_channel=[
        list(range(133)),  # 从 17 改为 133
    ],
    inference_channel=list(range(133)))  # 从 17 改为 133

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
        out_channels=channel_cfg['num_output_channels'], # 已对应修改
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

其中， `pretrained='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'` 表示采用 ImageNet 预训练的权重，初始化主干网络（backbone）。
不过，`pretrained` 只会初始化主干网络（backbone），而不会初始化网络头（head）。因此，我们模型微调时的预训练权重一般通过 `load_from` 指定，而不是使用 `pretrained` 指定。

## 支持自己的数据集

MMPose 支持十余种不同的数据集，包括 COCO, COCO-WholeBody, MPII, MPII-TRB 等数据集。
用户可将自定义数据集转换为已有数据集格式，并修改如下字段。

```python
data_root = 'data/coco'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCocoWholeBodyDataset', # 对应修改数据集名称
        ann_file=f'{data_root}/annotations/coco_wholebody_train_v1.0.json', # 修改数据集标签路径
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownCocoWholeBodyDataset', # 对应修改数据集名称
        ann_file=f'{data_root}/annotations/coco_wholebody_val_v1.0.json', # 修改数据集标签路径
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownCocoWholeBodyDataset', # 对应修改数据集名称
        ann_file=f'{data_root}/annotations/coco_wholebody_val_v1.0.json', # 修改数据集标签路径
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline)
)
```

## 修改训练策略

通常情况下，微调模型时设置较小的学习率和训练轮数，即可取得较好效果。

```python
# 优化器
optimizer = dict(
    type='Adam',
    lr=5e-4, # 可以适当减小
)
optimizer_config = dict(grad_clip=None)
# 学习策略
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200]) # 可以适当减小
total_epochs = 210 # 可以适当减小
```

## 使用预训练模型

网络设置中的 `pretrained`，仅会在主干网络模型上加载预训练参数。若要载入整个网络的预训练参数，需要通过 `load_from` 指定模型文件路径或模型链接。

```python
# 将预训练模型用于整个 HRNet 网络
load_from = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth'  # 模型路径可以在 model zoo 中找到
```
