_base_ = ['./yolox-pose_s_8xb32-300e_coco.py']

# model settings
model = dict(
    init_cfg=dict(checkpoint='https://download.openmmlab.com/mmyolo/v0/yolox/'
                  'yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco/yolox_tiny_fast_'
                  '8xb32-300e-rtmdet-hyp_coco_20230210_143637-4c338102.pth'),
    data_preprocessor=dict(batch_augments=[
        dict(
            type='PoseBatchSyncRandomResize',
            random_size_range=(320, 640),
            size_divisor=32,
            interval=1)
    ]),
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.375,
    ),
    neck=dict(
        deepen_factor=0.33,
        widen_factor=0.375,
    ),
    bbox_head=dict(head_module=dict(widen_factor=0.375)))

# data settings
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform

train_pipeline_stage1 = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=(img_scale),
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.75, 1.0),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='FilterDetPoseAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

test_pipeline = [
    *pre_transform,
    dict(type='mmdet.Resize', scale=(416, 416), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip_indices'))
]

train_dataloader = dict(
    batch_size=64, dataset=dict(pipeline=train_pipeline_stage1))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
