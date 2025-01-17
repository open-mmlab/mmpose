from mmengine.config import Config



def make_mmpose_config(
        data_root: str,
        classes: list,
        res: tuple = (192, 256),
):

    cfg = Config()

    cfg.default_scope = 'mmpose'
    cfg.gpu_ids = range(1)
    
    # hooks
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1, save_best='coco/AP', rule='greater'),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='PoseVisualizationHook', enable=True),
        badcase=dict(
            type='BadCaseAnalysisHook',
            enable=False,
            out_dir='badcase',
            metric_type='loss',
            badcase_thr=5))

    # custom hooks
    cfg.custom_hooks = [
        # Synchronize model buffers such as running_mean and running_var in BN
        # at the end of each epoch
        dict(type='SyncBuffersHook')
    ]

    # multi-processing backend
    cfg.env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl'),
    )

    # visualizer
    cfg.vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        # dict(type='WandbVisBackend'),
    ]
    cfg.visualizer = dict(
        type='PoseLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer')

    # logger
    cfg.log_processor = dict(
        type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
    cfg.log_level = 'INFO'
    cfg.load_from = None
    cfg.resume = False

    # file I/O backend
    cfg.backend_args = dict(backend='local')

    # training/validation/testing progress
    cfg.train_cfg = dict(max_epochs=300, val_interval=10, by_epoch=True)
    cfg.val_cfg = dict()
    cfg.test_cfg = dict()
    # runtime

    # optimizer
    cfg.optim_wrapper = dict(optimizer=dict(
        type='Adam',
        lr=5e-4,
    ))
    # learning policy
    cfg.param_scheduler = [
        dict(
            type='LinearLR', begin=0, end=500, start_factor=0.001,
            by_epoch=False),  # warm-up
        dict(
            type='MultiStepLR',
            begin=0,
            end=300,
            milestones=[200, 250],
            gamma=0.1,
            by_epoch=True)
    ]

    # automatically scaling LR based on the actual training batch size
    cfg.auto_scale_lr = dict(base_batch_size=64)

    # codec settings
    cfg.codec = dict(
        type='MSRAHeatmap', input_size=res, heatmap_size=(48, 64), sigma=2)

    # model settings
    cfg.model = dict(
        type='TopdownPoseEstimator',
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True),
        backbone=dict(
            type='ResNet',
            depth=18,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        ),
        head=dict(
            type='HeatmapHead',
            in_channels=512,
            out_channels=len(classes),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=cfg.codec),
        test_cfg=dict(
            flip_test=True,
            flip_mode='heatmap',
            shift_heatmap=True,
        ))

    # base dataset settings
    cfg.dataset_type = 'CocoDataset'
    cfg.data_mode = 'topdown'
    cfg.data_root = data_root

    # pipelines
    cfg.train_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='RandomBBoxTransform'),
        dict(type='TopdownAffine', input_size=cfg.codec['input_size']),
        dict(
            type='Albumentation',
            transforms=[
                dict(type='RandomBrightnessContrast', brightness_limit=[-0.2, 0.2], contrast_limit=[-0.2, 0.2], p=0.4),

                dict(
                    type='OneOf',
                    transforms=[
                        dict(type='MotionBlur', blur_limit=3, p=0.3),
                        dict(type='MedianBlur', blur_limit=3, p=0.2),
                        dict(type='Blur', blur_limit=3, p=0.2),
                    ], p=0.3),

                dict(
                    type='OneOf',
                    transforms=[
                        dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.3),
                        dict(type='MultiplicativeNoise', multiplier=(0.9, 1.1), p=0.3),
                    ], p=0.4),

                dict(type='HueSaturationValue', hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            ]),
        dict(type='GenerateTarget', encoder=cfg.codec),
        dict(type='PackPoseInputs')
    ]
    val_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=cfg.codec['input_size']),
        dict(type='PackPoseInputs')
    ]

    # data loaders
    cfg.train_dataloader = dict(
        batch_size=64,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type=cfg.dataset_type,
            labels=classes,
            data_root=data_root,
            data_mode=cfg.data_mode,
            ann_file='annotations/forklift_keypoints_train2017.json',
            data_prefix=dict(img='train2017/'),
            pipeline=cfg.train_pipeline,
        ))
    cfg.val_dataloader = dict(
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type=cfg.dataset_type,
            labels=classes,
            data_root=data_root,
            data_mode=cfg.data_mode,
            ann_file='annotations/forklift_keypoints_val2017.json',
            bbox_file='',
            data_prefix=dict(img='val2017/'),
            test_mode=True,
            pipeline=val_pipeline,
        ))
    cfg.test_dataloader = cfg.val_dataloader

    # evaluators
    cfg.val_evaluator = [
        dict(
            type='CocoMetric',
            ann_file=data_root + '/annotations/forklift_keypoints_val2017.json'
        ),
        dict(
            type='EPE',
        ),
        dict(
            type='PCKAccuracy',
            prefix="5pr_",
        ),
        dict(
            type='PCKAccuracy',
            thr=0.1,
            prefix="10pr_",
        ),
        dict(
            type='AUC',
        ),
    ]
    cfg.test_evaluator = cfg.val_evaluator
    return cfg