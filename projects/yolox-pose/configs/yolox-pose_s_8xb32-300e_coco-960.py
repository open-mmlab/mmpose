_base_ = 'mmyolo::yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py'

img_scale = (960, 960)
custom_imports = dict(imports=['models', 'datasets'])

# visualizer
visualizer = dict(
    type='mmpose.PoseLocalVisualizer',
    vis_backends=_base_.vis_backends,
    name='visualizer')

# model
model = dict(
    init_cfg=dict(
        _delete_=True,
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmyolo/v0/yolox/'
        'yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_'
        '8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth'),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        batch_augments=[
            dict(
                type='PoseBatchSyncRandomResize',
                random_size_range=(720, 1200),
                size_divisor=32,
                interval=1)
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
            loss_weight=10),
    ),
    test_cfg=dict(score_thr=0.1, multi_label=False))

# pipelines
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
    if 'img_scale' in transform:
        transform['img_scale'] = img_scale
    # if 'MixUp' in transform['type']:
    #     transform['prob'] = 0.1
    if transform['type'] == 'mmdet.RandomAffine':
        transform['scaling_ratio_range'] = (0.7, 1.1)
        transform['border'] = (-img_scale[0] // 2, -img_scale[1] // 2)
        # transform['max_translate_ratio'] = 0.1

train_pipeline_stage2 = [
    *pre_transform, *_base_.train_pipeline_stage2[2:-2],
    dict(type='FilterDetPoseAnnotations', keep_empty=False),
    dict(type='PackDetPoseInputs')
]

for transform in train_pipeline_stage2:
    if 'scale' in transform:
        transform['scale'] = img_scale

for hook in _base_.custom_hooks:
    if hook['type'] == 'YOLOXModeSwitchHook':
        hook['new_train_pipeline'] = train_pipeline_stage2

test_pipeline = [
    *pre_transform, *_base_.test_pipeline[1:-2],
    dict(
        type='PackDetPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip_indices'))
]
for transform in test_pipeline:
    if 'scale' in transform:
        transform['scale'] = img_scale

# dataset settings
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
)
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))
