# 教程 0: 模型配置文件

我们使用 python 文件作为配置文件，将模块化设计和继承设计结合到配置系统中，便于进行各种实验。
您可以在 `$MMPose/configs` 下找到所有提供的配置。如果要检查配置文件，您可以运行
`python tools/analysis/print_config.py /PATH/TO/CONFIG` 来查看完整的配置。

<!-- TOC -->

- [通过脚本参数修改配置](#通过脚本参数修改配置)
- [配置文件命名约定](#配置文件命名约定)
  - [配置系统](#配置系统)
- [常见问题](#常见问题)
  - [在配置中使用中间变量](#在配置中使用中间变量)

<!-- TOC -->

## 通过脚本参数修改配置

当使用 "tools/train.py" 或 "tools/test.py" 提交作业时，您可以指定 `--cfg-options` 来修改配置。

- 更新配置字典链的键值。

  可以按照原始配置文件中字典的键的顺序指定配置选项。
  例如，`--cfg-options model.backbone.norm_eval=False` 将主干网络中的所有 BN 模块更改为 `train` 模式。

- 更新配置列表内部的键值。

  一些配置字典在配置文件中会形成一个列表。例如，训练流水线 `data.train.pipeline` 通常是一个列表。
  例如，`[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]` 。如果要将流水线中的 `'flip_prob=0.5'` 更改为 `'flip_prob=0.0'`，您可以这样指定 `--cfg-options data.train.pipeline.1.flip_prob=0.0` 。

- 更新列表 / 元组的值。

  如果要更新的值是列表或元组，例如，配置文件通常设置为 `workflow=[('train', 1)]` 。
  如果您想更改这个键，您可以这样指定　`--cfg-options workflow="[(train,1),(val,1)]"` 。
  请注意，引号 " 是必要的，以支持列表 / 元组数据类型，并且指定值的引号内 **不允许** 有空格。

## 配置文件命名约定

我们按照下面的样式命名配置文件。建议贡献者也遵循同样的风格。

```
configs/{topic}/{task}/{algorithm}/{dataset}/{backbone}_[model_setting]_{dataset}_[input_size]_[technique].py
```

`{xxx}` 是必填字段，`[yyy]` 是可选字段.

- `{topic}`: 主题类型，如 `body`, `face`, `hand`, `animal` 等。
- `{task}`: 任务类型, `[2d | 3d]_[kpt | mesh]_[sview | mview]_[rgb | rgbd]_[img | vid]` 。任务类型从5个维度定义:（1）二维或三维姿态估计;（2）姿态表示形式:关键点 (kpt)、网格 (mesh) 或密集姿态 (dense); (3）单视图 (sview) 或多视图 (mview);（4）RGB 或 RGBD; 以及（5）图像 (img) 或视频 (vid)。例如， `2d_kpt_sview_rgb_img`，　`3d_kpt_sview_rgb_vid`,　等等。
- `{algorithm}`: 算法类型，例如，`associative_embedding`, `deeppose` 等。
- `{dataset}`: 数据集名称，例如， `coco` 等。
- `{backbone}`: 主干网络类型，例如，`res50` (ResNet-50) 等。
- `[model setting]`: 对某些模型的特定设置。
- `[input_size]`: 模型的输入大小。
- `[technique]`: 一些特定的技术，包括损失函数，数据增强，训练技巧等，例如， `wingloss`, `udp`, `fp16` 等.

### 配置系统

- 基于热图的二维自顶向下的人体姿态估计实例

  为了帮助用户对完整的配置结构和配置系统中的模块有一个基本的了解，
  我们下面对配置文件 'https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py' 作简要的注释。
  有关每个模块中每个参数的更详细用法和替代方法，请参阅 API 文档。

  ```python
  # 运行设置
  log_level = 'INFO'  # 日志记录级别
  load_from = None  # 从给定路径加载预训练模型
  resume_from = None  # 从给定路径恢复模型权重文件，将从保存模型权重文件时的轮次开始继续训练
  dist_params = dict(backend='nccl')  # 设置分布式训练的参数，也可以设置端口
  workflow = [('train', 1)]  # 运行程序的工作流。[('train', 1)] 表示只有一个工作流，名为 'train' 的工作流执行一次
  checkpoint_config = dict(  # 设置模型权重文件钩子的配置，请参阅 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py 的实现
      interval=10)  # 保存模型权重文件的间隔
  evaluation = dict(  # 训练期间评估的配置
      interval=10,  # 执行评估的间隔
      metric='mAP',  # 采用的评价指标
      key_indicator='AP')  # 将 `AP` 设置为关键指标以保存最佳模型权重文件
  # 优化器
  optimizer = dict(
      # 用于构建优化器的配置，支持 (1). PyTorch 中的所有优化器，
      # 其参数也与 PyTorch 中的相同. (2). 自定义的优化器
      # 它们通过 `constructor` 构建，可参阅 "tutorials/4_new_modules.md"
      # 的实现。
      type='Adam',  # 优化器的类型, 可参阅 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 获取更多细节
      lr=5e-4,  # 学习率, 参数的详细用法见 PyTorch 文档
  )
  optimizer_config = dict(grad_clip=None)  # 不限制梯度的范围
  # 学习率调整策略
  lr_config = dict(  # 用于注册 LrUpdater 钩子的学习率调度器的配置
      policy='step',  # 调整策略, 还支持 CosineAnnealing, Cyclic, 等等，请参阅 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9 获取支持的 LrUpdater 细节
      warmup='linear', # 使用的预热类型，它可以是 None (不使用预热), 'constant', 'linear' 或者 'exp'.
      warmup_iters=500,  # 预热的迭代次数或者轮数
      warmup_ratio=0.001,  # 预热开始时使用的学习率，等于预热比 (warmup_ratio) * 初始学习率
      step=[170, 200])  # 降低学习率的步数　
  total_epochs = 210  # 训练模型的总轮数
  log_config = dict(  # 注册日志记录器钩子的配置
      interval=50,  # 打印日志的间隔
      hooks=[
          dict(type='TextLoggerHook'),  # 用来记录训练过程的日志记录器
          # dict(type='TensorboardLoggerHook')  # 也支持 Tensorboard 日志记录器
      ])

  channel_cfg = dict(
      num_output_channels=17,  # 关键点头部的输出通道数
      dataset_joints=17,  # 数据集的关节数
      dataset_channel=[ # 数据集支持的通道数
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      ],
      inference_channel=[ # 输出通道数
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
      ])

  # 模型设置
  model = dict(  # 模型的配置
      type='TopDown',  # 模型的类型
      pretrained='torchvision://resnet50',  #　预训练模型的 url / 网址
      backbone=dict(  # 主干网络的字典
          type='ResNet',  # 主干网络的名称
          depth=50),  # ResNet 模型的深度
      keypoint_head=dict(  # 关键点头部的字典
          type='TopdownHeatmapSimpleHead',  # 关键点头部的名称
          in_channels=2048,  # 关键点头部的输入通道数
          out_channels=channel_cfg['num_output_channels'],  # 关键点头部的输出通道数
          loss_keypoint=dict(  # 关键点损失函数的字典
            type='JointsMSELoss',  # 关键点损失函数的名称
            use_target_weight=True)),  # 在损失计算中是否考虑目标权重
      train_cfg=dict(),  # 训练超参数的配置
      test_cfg=dict(  # 测试超参数的配置
          flip_test=True,  # 推断时是否使用翻转测试
          post_process='default',  # 使用“默认” (default) 后处理方法。
          shift_heatmap=True,  # 移动并对齐翻转的热图以获得更高的性能
          modulate_kernel=11))  # 用于调制的高斯核大小。仅用于 "post_process='unbiased'"

  data_cfg = dict(
      image_size=[192, 256],  # 模型输入分辨率的大小
      heatmap_size=[48, 64],  # 输出热图的大小
      num_output_channels=channel_cfg['num_output_channels'],  # 输出通道数
      num_joints=channel_cfg['dataset_joints'],  # 关节点数量
      dataset_channel=channel_cfg['dataset_channel'], # 数据集支持的通道数
      inference_channel=channel_cfg['inference_channel'], # 输出通道数
      soft_nms=False,  # 推理过程中是否执行 soft_nms
      nms_thr=1.0,  # 非极大抑制阈值
      oks_thr=0.9,  # nms 期间 oks（对象关键点相似性）得分阈值
      vis_thr=0.2,  # 关键点可见性阈值
      use_gt_bbox=False,  # 测试时是否使用人工标注的边界框
      det_bbox_thr=0.0,  # 检测到的边界框分数的阈值。当 'use_gt_bbox=True' 时使用
      bbox_file='data/coco/person_detection_results/'  # 边界框检测文件的路径
      'COCO_val2017_detections_AP_H_56_person.json',
  )

  train_pipeline = [
      dict(type='LoadImageFromFile'),  # 从文件加载图像
      dict(type='TopDownRandomFlip',  # 执行随机翻转增强
           flip_prob=0.5),  # 执行翻转的概率
      dict(
          type='TopDownHalfBodyTransform',  # TopDownHalfBodyTransform 数据增强的配置
          num_joints_half_body=8,  # 执行半身变换的阈值
          prob_half_body=0.3),  # 执行翻转的概率
      dict(
          type='TopDownGetRandomScaleRotation',   #　TopDownGetRandomScaleRotation 的配置
          rot_factor=40,  # 旋转到 ``[-2*rot_factor, 2*rot_factor]``.
          scale_factor=0.5), # 缩放到 ``[1-scale_factor, 1+scale_factor]``.
      dict(type='TopDownAffine',  # 对图像进行仿射变换形成输入
          use_udp=False),  # 不使用无偏数据处理
      dict(type='ToTensor'),  # 将其他类型转换为张量类型流水线
      dict(
          type='NormalizeTensor',  # 标准化输入张量
          mean=[0.485, 0.456, 0.406],  # 要标准化的不同通道的平均值
          std=[0.229, 0.224, 0.225]),  # 要标准化的不同通道的标准差
      dict(type='TopDownGenerateTarget',  # 生成热图目标。支持不同的编码类型
           sigma=2),  # 热图高斯的 Sigma
      dict(
          type='Collect',  # 收集决定数据中哪些键应该传递到检测器的流水线
          keys=['img', 'target', 'target_weight'],  # 输入键
          meta_keys=[  # 输入的元键
              'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
              'rotation', 'bbox_score', 'flip_pairs'
          ]),
  ]

  val_pipeline = [
      dict(type='LoadImageFromFile'),  # 从文件加载图像
      dict(type='TopDownAffine'),  # 对图像进行仿射变换形成输入
      dict(type='ToTensor'),  # ToTensor 的配置
      dict(
          type='NormalizeTensor',
          mean=[0.485, 0.456, 0.406],  # 要标准化的不同通道的平均值
          std=[0.229, 0.224, 0.225]),  # 要标准化的不同通道的标准差
      dict(
          type='Collect',  # 收集决定数据中哪些键应该传递到检测器的流水线
          keys=['img'],  # 输入键
          meta_keys=[  # 输入的元键
              'image_file', 'center', 'scale', 'rotation', 'bbox_score',
              'flip_pairs'
          ]),
  ]

  test_pipeline = val_pipeline

  data_root = 'data/coco'  # 数据集的配置
  data = dict(
      samples_per_gpu=64,  # 训练期间每个 GPU 的 Batch size
      workers_per_gpu=2,  # 每个 GPU 预取数据的 worker 个数
      val_dataloader=dict(samples_per_gpu=32),  # 验证期间每个 GPU 的 Batch size
      test_dataloader=dict(samples_per_gpu=32),  # 测试期间每个 GPU 的 Batch size
      train=dict(  # 训练数据集的配置
          type='TopDownCocoDataset',  # 数据集的名称
          ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',  # 标注文件的路径
          img_prefix=f'{data_root}/train2017/',
          data_cfg=data_cfg,
          pipeline=train_pipeline),
      val=dict(  # 验证数据集的配置
          type='TopDownCocoDataset',  # 数据集的名称
          ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',  # 标注文件的路径
          img_prefix=f'{data_root}/val2017/',
          data_cfg=data_cfg,
          pipeline=val_pipeline),
      test=dict(  # 测试数据集的配置
          type='TopDownCocoDataset',  # 数据集的名称
          ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',  # 标注文件的路径
          img_prefix=f'{data_root}/val2017/',
          data_cfg=data_cfg,
          pipeline=val_pipeline),
  )

  ```

## 常见问题

### 在配置中使用中间变量

配置文件中使用了一些中间变量，如 `train_pipeline`/`val_pipeline`/`test_pipeline` 等。

例如，我们首先要定义 `train_pipeline`/`val_pipeline`/`test_pipeline`，然后将它们传递到 `data` 中。
因此，`train_pipeline`/`val_pipeline`/`test_pipeline` 是中间变量。
