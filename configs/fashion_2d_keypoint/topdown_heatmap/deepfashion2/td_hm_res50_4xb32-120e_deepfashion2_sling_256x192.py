_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/datasets/deepfashion2.py'
]

default_hooks = dict(checkpoint=dict(save_best='PCK', rule='greater'))

resume = False  # 断点恢复
load_from = None  # 模型权重加载
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=10)  # 训练轮数，测试间隔
param_scheduler = [
    dict(  # warmup策略
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.001,
        by_epoch=False),
    dict(  # scheduler
        type='MultiStepLR',
        begin=0,
        end=120,
        milestones=[80, 100],
        gamma=0.1,
        by_epoch=True)
]
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))  # 优化器和学习率
auto_scale_lr = dict(base_batch_size=512)  # 根据batch_size自动缩放学习率

backend_args = dict(backend='local')  # 数据加载后端设置，默认从本地硬盘加载
dataset_type = 'DeepFashion2Dataset'  # 数据集类名  DeepFashionDataset
data_mode = 'topdown'  # 算法结构类型，用于指定标注信息加载策略
data_root = 'data/deepfashion2/'  # 数据存放路径
# 定义数据编解码器，用于生成target和对pred进行解码，同时包含了输入图片和输出heatmap尺寸等信息
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=60,
        scale_factor=(0.75, 1.25)),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [  # 测试时数据增强
    dict(type='LoadImage', backend_args=backend_args),  # 加载图片
    dict(type='GetBBoxCenterScale'),  # 根据bbox获取center和scale
    dict(type='TopdownAffine', input_size=codec['input_size']),  # 根据变换矩阵更新目标数据
    dict(type='PackPoseInputs')  # 对target进行打包用于训练
]
train_dataloader = dict(  # 训练数据加载
    batch_size=32,  # 批次大小
    num_workers=6,  # 数据加载进程数
    persistent_workers=True,  # 在不活跃时维持进程不终止，避免反复启动进程的开销
    sampler=dict(type='DefaultSampler', shuffle=True),  # 采样策略，打乱数据
    dataset=dict(
        type=dataset_type,  # 数据集类名
        data_root=data_root,  # 数据集路径
        data_mode=data_mode,  # 算法类型
        ann_file='train/deepfashion2_sling_train.json',  # 标注文件路径
        data_prefix=dict(img='train/image/'),  # 图像路径
        pipeline=train_pipeline  # 数据流水线
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=6,
    persistent_workers=True,  # 在不活跃时维持进程不终止，避免反复启动进程的开销
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),  # 采样策略，不进行打乱
    dataset=dict(
        type=dataset_type,  # 数据集类名
        data_root=data_root,  # 数据集路径
        data_mode=data_mode,  # 算法类型
        ann_file='validation/deepfashion2_sling_validation.json',  # 标注文件路径
        data_prefix=dict(img='validation/image/'),  # 图像路径
        test_mode=True,  # 测试模式开关
        pipeline=val_pipeline  # 数据流水线
    ))
test_dataloader = val_dataloader  # 默认情况下不区分验证集和测试集，用户根据需要来自行定义

channel_cfg = dict(
    num_output_channels=294,
    dataset_joints=294,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
            103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
            116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
            129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
            142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
            155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
            168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
            181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
            194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
            207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
            220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
            233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
            246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
            259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
            272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284,
            285, 286, 287, 288, 289, 290, 291, 292, 293
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
        92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
        122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
        150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
        164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
        178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
        206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
        220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
        234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
        248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261,
        262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
        276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
        290, 291, 292, 293
    ])

model = dict(
    type='TopdownPoseEstimator',  # 模型结构决定了算法流程
    data_preprocessor=dict(  # 数据归一化和通道顺序调整，作为模型的一部分
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(
            type='Pretrained',  # 预训练参数，只加载backbone权重用于迁移学习
            checkpoint='torchvision://resnet50')),
    head=dict(  # 模型头部
        type='HeatmapHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
        # deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),  # 损失函数
        decoder=codec),  # 解码器，将heatmap解码成坐标值
    test_cfg=dict(
        flip_test=True,  # 开启测试时水平翻转集成
        flip_mode='heatmap',  # 对heatmap进行翻转
        shift_heatmap=True,  # 对翻转后的结果进行平移提高精度
    ))

val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE'),
]
test_evaluator = val_evaluator  # 默认情况下不区分验证集和测试集，用户根据需要来自行定义

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend')])
