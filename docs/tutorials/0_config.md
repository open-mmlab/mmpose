# Tutorial 0: Learn about Configs

We use python files as configs, incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
You can find all the provided configs under `$MMPose/configs`. If you wish to inspect the config file,
you may run `python tools/analysis/print_config.py /PATH/TO/CONFIG` to see the complete config.

<!-- TOC -->

- [Modify config through script arguments](#modify-config-through-script-arguments)
- [Config File Naming Convention](#config-file-naming-convention)
  - [Config System Example](#config-system-example)
- [FAQ](#faq)
  - [Use intermediate variables in configs](#use-intermediate-variables-in-configs)

<!-- TOC -->

## Modify config through script arguments

When submitting jobs using "tools/train.py" or "tools/test.py", you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]`. If you want to change `'flip_prob=0.5'` to `'flip_prob=0.0'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.1.flip_prob=0.0`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark \" is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Config File Naming Convention

We follow the style below to name config files. Contributors are advised to follow the same style.

```
configs/{topic}/{task}/{algorithm}/{dataset}/{backbone}_[model_setting]_{dataset}_[input_size]_[technique].py
```

`{xxx}` is required field and `[yyy]` is optional.

- `{topic}`: topic type, e.g. `body`, `face`, `hand`, `animal`, etc.
- `{task}`: task type, `[2d | 3d]_[kpt | mesh]_[sview | mview]_[rgb | rgbd]_[img | vid]`. The task is categorized in 5: (1) 2D or 3D pose estimation, (2) representation type: keypoint (kpt), mesh, or DensePose (dense). (3) Single-view (sview) or multi-view (mview), (4) RGB or RGBD, and (5) Image (img) or Video (vid). e.g. `2d_kpt_sview_rgb_img`, `3d_kpt_sview_rgb_vid`, etc.
- `{algorithm}`: algorithm type, e.g. `associative_embedding`, `deeppose`, etc.
- `{dataset}`: dataset name, e.g. `coco`, etc.
- `{backbone}`: backbone type, e.g. `res50` (ResNet-50), etc.
- `[model setting]`: specific setting for some models.
- `[input_size]`: input size of the model.
- `[technique]`: some specific techniques, including losses, augmentation and tricks, e.g. `wingloss`, `udp`, `fp16`.

### Config System

- An Example of 2D Top-down Heatmap-based Human Pose Estimation

    To help the users have a basic idea of a complete config structure and the modules in the config system,
    we make brief comments on 'https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py' as the following.
    For more detailed usage and alternative for per parameter in each module, please refer to the API documentation.

    ```python
    # runtime settings
    log_level = 'INFO'  # The level of logging
    load_from = None  # load models as a pre-trained model from a given path. This will not resume training
    resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
    dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
    workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once
    checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation
        interval=10)  # Interval to save checkpoint
    evaluation = dict(  # Config of evaluation during training
        interval=10,  # Interval to perform evaluation
        metric='mAP',  # Metrics to be performed
        save_best='AP')  # set `AP` as key indicator to save best checkpoint
    # optimizer
    optimizer = dict(
        # Config used to build optimizer, support (1). All the optimizers in PyTorch
        # whose arguments are also the same as those in PyTorch. (2). Custom optimizers
        # which are builed on `constructor`, referring to "tutorials/4_new_modules.md"
        # for implementation.
        type='Adam',  # Type of optimizer, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
        lr=5e-4,  # Learning rate, see detail usages of the parameters in the documentation of PyTorch
    )
    optimizer_config = dict(grad_clip=None)  # Do not use gradient clip
    # learning policy
    lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
        policy='step',  # Policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
        warmup='linear', # Type of warmup used. It can be None(use no warmup), 'constant', 'linear' or 'exp'.
        warmup_iters=500,  # The number of iterations or epochs that warmup
        warmup_ratio=0.001,  # LR used at the beginning of warmup equals to warmup_ratio * initial_lr
        step=[170, 200])  # Steps to decay the learning rate
    total_epochs = 210  # Total epochs to train the model
    log_config = dict(  # Config to register logger hook
        interval=50,  # Interval to print the log
        hooks=[
            dict(type='TextLoggerHook'),  # The logger used to record the training process
            # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        ])

    channel_cfg = dict(
        num_output_channels=17,  # The output channels of keypoint head
        dataset_joints=17,  # Number of joints in the dataset
        dataset_channel=[ # Dataset supported channels
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ],
        inference_channel=[ # Channels to output
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ])

    # model settings
    model = dict(  # Config of the model
        type='TopDown',  # Type of the model
        pretrained='torchvision://resnet50',  # The url/site of the pretrained model
        backbone=dict(  # Dict for backbone
            type='ResNet',  # Name of the backbone
            depth=50),  # Depth of ResNet model
        keypoint_head=dict(  # Dict for keypoint head
            type='TopdownHeatmapSimpleHead',  # Name of keypoint head
            in_channels=2048,  # The input channels of keypoint head
            out_channels=channel_cfg['num_output_channels'],  # The output channels of keypoint head
            loss_keypoint=dict(  # Dict for keypoint loss
              type='JointsMSELoss',  # Name of keypoint loss
              use_target_weight=True)),  # Whether to consider target_weight during loss calculation
        train_cfg=dict(),  # Config of training hyper-parameters
        test_cfg=dict(  # Config of testing hyper-parameters
            flip_test=True,  # Whether to use flip-test during inference
            post_process='default',  # Use 'default' post-processing approach.
            shift_heatmap=True,  # Shift and align the flipped heatmap to achieve higher performance
            modulate_kernel=11))  # Gaussian kernel size for modulation. Only used for "post_process='unbiased'"

    data_cfg = dict(
        image_size=[192, 256],  # Size of model input resolution
        heatmap_size=[48, 64],  # Size of the output heatmap
        num_output_channels=channel_cfg['num_output_channels'],  # Number of output channels
        num_joints=channel_cfg['dataset_joints'],  # Number of joints
        dataset_channel=channel_cfg['dataset_channel'], # Dataset supported channels
        inference_channel=channel_cfg['inference_channel'], # Channels to output
        soft_nms=False,  # Whether to perform soft-nms during inference
        nms_thr=1.0,  # Threshold for non maximum suppression.
        oks_thr=0.9,  # Threshold of oks (object keypoint similarity) score during nms
        vis_thr=0.2,  # Threshold of keypoint visibility
        use_gt_bbox=False,  # Whether to use ground-truth bounding box during testing
        det_bbox_thr=0.0,  # Threshold of detected bounding box score. Used when 'use_gt_bbox=True'
        bbox_file='data/coco/person_detection_results/'  # Path to the bounding box detection file
        'COCO_val2017_detections_AP_H_56_person.json',
    )

    train_pipeline = [
        dict(type='LoadImageFromFile'),  # Loading image from file
        dict(type='TopDownRandomFlip',  # Perform random flip augmentation
             flip_prob=0.5),  # Probability of implementing flip
        dict(
            type='TopDownHalfBodyTransform',  # Config of TopDownHalfBodyTransform data-augmentation
            num_joints_half_body=8,  # Threshold of performing half-body transform.
            prob_half_body=0.3),  # Probability of implementing half-body transform
        dict(
            type='TopDownGetRandomScaleRotation',   # Config of TopDownGetRandomScaleRotation
            rot_factor=40,  # Rotating to ``[-2*rot_factor, 2*rot_factor]``.
            scale_factor=0.5), # Scaling to ``[1-scale_factor, 1+scale_factor]``.
        dict(type='TopDownAffine',  # Affine transform the image to make input.
            use_udp=False),  # Do not use unbiased data processing.
        dict(type='ToTensor'),  # Convert other types to tensor type pipeline
        dict(
            type='NormalizeTensor',  # Normalize input tensors
            mean=[0.485, 0.456, 0.406],  # Mean values of different channels to normalize
            std=[0.229, 0.224, 0.225]),  # Std values of different channels to normalize
        dict(type='TopDownGenerateTarget',  # Generate heatmap target. Different encoding types supported.
             sigma=2),  # Sigma of heatmap gaussian
        dict(
            type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the detector
            keys=['img', 'target', 'target_weight'],  # Keys of input
            meta_keys=[  # Meta keys of input
                'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
                'rotation', 'bbox_score', 'flip_pairs'
            ]),
    ]

    val_pipeline = [
        dict(type='LoadImageFromFile'),  # Loading image from file
        dict(type='TopDownAffine'),  # Affine transform the image to make input.
        dict(type='ToTensor'),  # Config of ToTensor
        dict(
            type='NormalizeTensor',
            mean=[0.485, 0.456, 0.406],  # Mean values of different channels to normalize
            std=[0.229, 0.224, 0.225]),  # Std values of different channels to normalize
        dict(
            type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the detector
            keys=['img'],  # Keys of input
            meta_keys=[  # Meta keys of input
                'image_file', 'center', 'scale', 'rotation', 'bbox_score',
                'flip_pairs'
            ]),
    ]

    test_pipeline = val_pipeline

    data_root = 'data/coco' # Root of the dataset
    data = dict( # Config of data
        samples_per_gpu=64,  # Batch size of each single GPU during training
        workers_per_gpu=2,  # Workers to pre-fetch data for each single GPU
        val_dataloader=dict(samples_per_gpu=32),  # Batch size of each single GPU during validation
        test_dataloader=dict(samples_per_gpu=32),  # Batch size of each single GPU during testing
        train=dict(  # Training dataset config
            type='TopDownCocoDataset',  # Name of dataset
            ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',  # Path to annotation file
            img_prefix=f'{data_root}/train2017/',
            data_cfg=data_cfg,
            pipeline=train_pipeline),
        val=dict(  # Validation dataset config
            type='TopDownCocoDataset',  # Name of dataset
            ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',  # Path to annotation file
            img_prefix=f'{data_root}/val2017/',
            data_cfg=data_cfg,
            pipeline=val_pipeline),
        test=dict(  # Testing dataset config
            type='TopDownCocoDataset',  # Name of dataset
            ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',  # Path to annotation file
            img_prefix=f'{data_root}/val2017/',
            data_cfg=data_cfg,
            pipeline=val_pipeline),
    )

    ```

## FAQ

### Use intermediate variables in configs

Some intermediate variables are used in the config files, like `train_pipeline`/`val_pipeline`/`test_pipeline` etc.

For Example, we would like to first define `train_pipeline`/`val_pipeline`/`test_pipeline` and pass them into `data`.
Thus, `train_pipeline`/`val_pipeline`/`test_pipeline` are intermediate variable.
