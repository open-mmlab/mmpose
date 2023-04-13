# Configs

We use python files as configs and incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.

## Introduction

MMPose is equipped with a powerful config system. Cooperating with Registry, a config file can organize all the configurations in the form of python dictionaries and create instances of the corresponding modules.

Here is a simple example of vanilla Pytorch module definition to show how the config system works:

```Python
# Definition of Loss_A in loss_a.py
Class Loss_A(nn.Module):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    def forward(self, x):
        return x

# Init the module
loss = Loss_A(param1=1.0, param2=True)
```

All you need to do is just to register the module to the pre-defined Registry `MODELS`:

```Python
# Definition of Loss_A in loss_a.py
from mmpose.registry import MODELS

@MODELS.register_module() # register the module to MODELS
Class Loss_A(nn.Module):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    def forward(self, x):
        return x
```

And import the new module in `__init__.py` in the corresponding directory:

```Python
# __init__.py of mmpose/models/losses
from .loss_a.py import Loss_A

__all__ = ['Loss_A']
```

Then you can define the module anywhere you want：

```Python
# config_file.py
loss_cfg = dict(
    type='Loss_A', # specify your registered module via `type`
    param1=1.0,    # pass parameters to __init__() of the module
    param2=True
)

# Init the module
loss = MODELS.build(loss_cfg) # equals to `loss = Loss_A(param1=1.0, param2=True)`
```

```{note}
Note that all new modules need to be registered using `Registry` and imported in `__init__.py` in the corresponding directory before we can create their instances from configs.
```

Here is a list of pre-defined registries in MMPose:

- `DATASETS`: data-related modules
- `TRANSFORMS`: data transformations
- `MODELS`: all kinds of modules inheriting `nn.Module` (Backbone, Neck, Head, Loss, etc.)
- `VISUALIZERS`: visualization tools
- `VISBACKENDS`: visualizer backend
- `METRICS`: all kinds of evaluation metrics
- `KEYPOINT_CODECS`: keypoint encoder/decoder
- `HOOKS`: all kinds of hooks like `CheckpointHook`

All registries are defined in `$MMPOSE/mmpose/registry.py`.

## Config System

It is best practice to layer your configs in five sections:

- **General**: basic configurations non-related to training or testing, such as Timer, Logger, Visualizer and other Hooks, as well as distributed-related environment settings

- **Data**: dataset, dataloader and data augmentation

- **Training**: resume, weights loading, optimizer, learning rate scheduling, epochs and valid interval etc.

- **Model**: structure, module and loss function etc.

- **Evaluation**: metrics

You can find all the provided configs under `$MMPOSE/configs`. A config can inherit contents from another config.To keep a config file simple and easy to read, we store some necessary but unremarkable configurations to `$MMPOSE/configs/_base_`.You can inspect the complete configurations by：

```Bash
python tools/analysis/print_config.py /PATH/TO/CONFIG
```

### General

General configuration refers to the necessary configuration non-related to training or testing, mainly including:

- **Default Hooks**: time statistics, training logs, checkpoints etc.

- **Environment**: distributed backend, cudnn, multi-processing etc.

- **Visualizer**: visualization backend and strategy

- **Log**: log level, format, printing and recording interval etc.

Here is the description of General configuration:

```Python
# General
default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'), # time the data processing and model inference
    logger=dict(type='LoggerHook', interval=50), # interval to print logs
    param_scheduler=dict(type='ParamSchedulerHook'), # update lr
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='coco/AP', # interval to save ckpt
        rule='greater'), # rule to judge the metric
    sampler_seed=dict(type='DistSamplerSeedHook')) # set the distributed seed
env_cfg = dict(
    cudnn_benchmark=False, # cudnn benchmark flag
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # num of opencv threads
    dist_cfg=dict(backend='nccl')) # distributed training backend
vis_backends = [dict(type='LocalVisBackend')] # visualizer backend
visualizer = dict( # Config of visualizer
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict( # Format, interval to log
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO' # The level of logging
```

General configuration is stored alone in the `$MMPOSE/configs/_base_`, and inherited by doing:

```Python
_base_ = ['../../../_base_/default_runtime.py'] # take the config file as the starting point of the relative path
```

```{note}
CheckpointHook:

- save_best: `'coco/AP'` for `CocoMetric`, `'PCK'` for `PCKAccuracy`
- max_keep_ckpts: the maximum checkpoints to keep. Defaults to -1, which means unlimited.

Example:

`default_hooks = dict(checkpoint=dict(save_best='PCK', rule='greater', max_keep_ckpts=1))`
```

### Data

Data configuration refers to the data processing related settings, mainly including:

- **File Client**: data storage backend, default is `disk`, we also support `LMDB`, `S3 Bucket` etc.

- **Dataset**: image and annotation file path

- **Dataloader**: loading configuration, batch size etc.

- **Pipeline**: data augmentation

- **Input Encoder**: encoding the annotation into specific form of target

Here is the description of Data configuration:

```Python
backend_args = dict(backend='local') # data storage backend
dataset_type = 'CocoDataset' # name of dataset
data_mode = 'topdown' # type of the model
data_root = 'data/coco/' # root of the dataset
 # config of codec，to generate targets and decode preds into coordinates
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
train_pipeline = [ # data aug in training
    dict(type='LoadImage', backend_args=backend_args, # image loading
    dict(type='GetBBoxCenterScale'), # calculate center and scale of bbox
    dict(type='RandomBBoxTransform'), # config of scaling, rotation and shifing
    dict(type='RandomFlip', direction='horizontal'), # config of random flipping
    dict(type='RandomHalfBody'), # config of half-body aug
    dict(type='TopdownAffine', input_size=codec['input_size']), # update inputs via transform matrix
    dict(
        type='GenerateTarget', # generate targets via transformed inputs
        # typeof targets
        encoder=codec, # get encoder from codec
    dict(type='PackPoseInputs') # pack targets
]
test_pipeline = [ # data aug in testing
    dict(type='LoadImage', backend_args=backend_args), # image loading
    dict(type='GetBBoxCenterScale'), # calculate center and scale of bbox
    dict(type='TopdownAffine', input_size=codec['input_size']), # update inputs via transform matrix
    dict(type='PackPoseInputs') # pack targets
]
train_dataloader = dict(
    batch_size=64, # batch size of each single GPU during training
    num_workers=2, # workers to pre-fetch data for each single GPU
    persistent_workers=True, # workers will stay around (with their state) waiting for another call into that dataloader.
    sampler=dict(type='DefaultSampler', shuffle=True), # data sampler, shuffle in traning
    dataset=dict(
        type=dataset_type , # name of dataset
        data_root=data_root, # root of dataset
        data_mode=data_mode, # type of the model
        ann_file='annotations/person_keypoints_train2017.json', # path to annotation file
        data_prefix=dict(img='train2017/'), # path to images
        pipeline=train_pipeline
    ))
val_dataloader = dict(
    batch_size=32, # batch size of each single GPU during validation
    num_workers=2, # workers to pre-fetch data for each single GPU
    persistent_workers=True, # workers will stay around (with their state) waiting for another call into that dataloader.
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False), # data sampler
    dataset=dict(
        type=dataset_type , # name of dataset
        data_root=data_root, # root of dataset
        data_mode=data_mode, # type of the model
        ann_file='annotations/person_keypoints_val2017.json', # path to annotation file
        bbox_file=
        'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json', # bbox file use for evaluation
        data_prefix=dict(img='val2017/'), # path to images
        test_mode=True,
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader # use val as test by default
```

```{note}
Common Usages:
- [Resume training](../common_usages/resume_training.md)
- [Automatic mixed precision (AMP) training](../common_usages/amp_training.md)
- [Set the random seed](../common_usages/set_random_seed.md)

```

### Training

Training configuration refers to the training related settings including:

- Resume training

- Model weights loading

- Epochs of training and interval to validate

- Learning rate adjustment strategies like warm-up, scheduling etc.

- Optimizer and initial learning rate

- Advanced tricks like auto learning rate scaling

Here is the description of Training configuration:

```Python
resume = False # resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
load_from = None # load models as a pre-trained model from a given path
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10) # max epochs of training, interval to validate
param_scheduler = [
    dict( # warmup strategy
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict( # scheduler
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005)) # optimizer and initial lr
auto_scale_lr = dict(base_batch_size=512) # auto scale the lr according to batch size
```

### Model

Model configuration refers to model training and inference related settings including:

- Model Structure

- Loss Function

- Output Decoding

- Test-time augmentation

Here is the description of Model configuration, which defines a Top-down Heatmap-based HRNetx32:

```Python
# config of codec, if already defined in data configuration section, no need to define again
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

model = dict(
    type='TopdownPoseEstimator', # Macro model structure
    data_preprocessor=dict( # data normalization and channel transposition
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict( # config of backbone
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
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained', # load pretrained weights to backbone
            checkpoint='https://download.openmmlab.com/mmpose'
            '/pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict( # config of head
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True), # config of loss function
        decoder=codec), # get decoder from codec
    test_cfg=dict(
        flip_test=True, # flag of flip test
        flip_mode='heatmap', # heatmap flipping
        shift_heatmap=True,  # shift the flipped heatmap several pixels to get a better performance
    ))
```

### Evaluation

Evaluation configuration refers to metrics commonly used by public datasets for keypoint detection tasks, mainly including:

- AR, AP and mAP

- PCK, PCKh, tPCK

- AUC

- EPE

- NME

Here is the description of Evaluation configuration, which defines a COCO metric evaluator:

```Python
val_evaluator = dict(
    type='CocoMetric', # coco AP
    ann_file=data_root + 'annotations/person_keypoints_val2017.json') # path to annotation file
test_evaluator = val_evaluator # use val as test by default
```

## Config File Naming Convention

MMPose follow the style below to name config files：

```Python
{{algorithm info}}_{{module info}}_{{training info}}_{{data info}}.py
```

The filename is divided into four parts:

- **Algorithm Information**: the name of algorithm, such as `topdown-heatmap`, `topdown-rle`

- **Module Information**: list of intermediate modules in the forward order, such as `res101`, `hrnet-w48`

- **Training Information**: settings of training(e.g. `batch_size`, `scheduler`), such as `8xb64-210e`

- **Data Information**: the name of dataset, the reshape of input data, such as `ap10k-256x256`, `zebra-160x160`

Words between different parts are connected by `'_'`, and those from the same part are connected by `'-'`.

To avoid a too long filename, some strong related modules in `{{module info}}` will be omitted, such as `gap` in `RLE` algorithm, `deconv` in `Heatmap-based` algorithm

Contributors are advised to follow the same style.

## Common Usage

### Inheritance

This is often used to inherit configurations from other config files. Let's assume two configs like:

`optimizer_cfg.py`:

```Python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

`resnet50.py`:

```Python
_base_ = ['optimizer_cfg.py']
model = dict(type='ResNet', depth=50)
```

Although we did not define `optimizer` in `resnet50.py`, all configurations in `optimizer.py` will be inherited by setting `_base_ = ['optimizer_cfg.py']`

```Python
cfg = Config.fromfile('resnet50.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

### Modification

For configurations already set in previous configs, you can directly modify arguments specific to that module.

`resnet50_lr0.01.py`:

```Python
_base_ = ['optimizer_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(lr=0.01) # modify specific filed
```

Now only `lr` is modified:

```Python
cfg = Config.fromfile('resnet50_lr0.01.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

### Delete

For configurations already set in previous configs, if you wish to modify some specific argument and delete the remainders(in other words, discard the previous and redefine the module), you can set `_delete_=True`.

`resnet50.py`:

```Python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(_delete_=True, type='SGD', lr=0.01) # discard the previous and redefine the module
```

Now only `type` and `lr` are kept:

```Python
cfg = Config.fromfile('resnet50_lr0.01.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.01)
```

```{note}
If you wish to learn more about advanced usages of the config system, please refer to [MMEngine Config](https://mmengine.readthedocs.io/en/latest/tutorials/config.html).
```
