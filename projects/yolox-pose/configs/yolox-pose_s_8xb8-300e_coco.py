# Directly inherit the entire recipe you want to use.
_base_ = 'mmyolo::yolox/yolox_s_fast_8xb8-300e_coco.py'

# This line is to import your own modules.
custom_imports = dict(imports=['models', 'datasets'])

# Modify the model to use your own head and loss.
model = dict(
    init_cfg=dict(
        _delete_=True,
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmyolo/'
        'v0/yolox/yolox_s_8xb8-300e_coco/'
        'yolox_s_8xb8-300e_coco_20220917_030738-d7e60cb2.pth'),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        batch_augments=[
            dict(
                type='PoseBatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
        ]),
    bbox_head=dict(
        type='YOLOXPoseHead',
        head_module=dict(
            type='YOLOXPoseHeadModule',
            num_classes=1,
            num_keypoints=17,
        ),
        loss_pose=dict(
            type='OksLoss',
            metainfo='configs/_base_/datasets/coco.py',
            loss_weight=50),
    ),
    test_cfg=dict(multi_label=False))

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='PoseToDetConverter')
]

train_pipeline_stage1 = [
    *pre_transform, *_base_.train_pipeline_stage1[2:-2],
    dict(type='FilterDetPoseAnnotations', keep_empty=False),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

for transform in train_pipeline_stage1:
    if 'pre_transform' in transform:
        transform['pre_transform'] = pre_transform

train_pipeline_stage2 = [
    *pre_transform, *_base_.train_pipeline_stage2[2:-2],
    dict(type='FilterDetPoseAnnotations', keep_empty=False),
    dict(type='PackDetPoseInputs')
]

test_pipeline = [
    *pre_transform, *_base_.test_pipeline[1:-2],
    dict(
        type='PackDetPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip_indices'))
]

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'bottomup'
data_root = 'data/coco/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        pipeline=train_pipeline_stage1))
_base_.train_dataloader.pop('collate_fn')

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        test_mode=True,
        ann_file='annotations/person_keypoints_val2017.json',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    _delete_=True,
    type='mmpose.CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    nms_mode='none',
    # score_mode='keypoint',
)
test_evaluator = val_evaluator

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))
