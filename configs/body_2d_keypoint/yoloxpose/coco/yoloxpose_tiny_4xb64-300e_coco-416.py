_base_ = './yoloxpose_s_8xb32-300e_coco-640.py'

# model settings
widen_factor = 0.375
deepen_factor = 0.33
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_' \
    'tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            random_size_range=(320, 640),
            size_divisor=32,
            interval=1),
    ]),
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint),
    ),
    neck=dict(
        in_channels=[96, 192, 384],
        out_channels=96,
    ),
    head=dict(head_module_cfg=dict(widen_factor=widen_factor), ))

# dataset settings
train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=_base_.input_size,
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=_base_.input_size,
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=_base_.codec),
    dict(
        type='PackPoseInputs',
        extra_mapping_labels={
            'bbox': 'bboxes',
            'bbox_labels': 'labels',
            'keypoints': 'keypoints',
            'keypoints_visible': 'keypoints_visible',
            'area': 'areas'
        }),
]
train_dataloader = dict(
    batch_size=64, dataset=dict(pipeline=train_pipeline_stage1))

input_size = (416, 416)
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]

val_dataloader = dict(dataset=dict(pipeline=val_pipeline, ))
test_dataloader = val_dataloader
