_base_ = ['hrnet_w32_coco_512x512.py']

model = dict(
    test_cfg=dict(
        multi_scale_score_decrease=1.0,
        nms_dist_thr=0.1,
        max_pool_kernel=9,
    ))

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpGetImgSize',
        base_length=32,
        test_scale_factor=[0.5, 1, 2]),
    dict(
        type='BottomUpResizeAlign',
        base_length=32,
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index', 'num_joints', 'skeleton',
            'image_size', 'heatmap_size'
        ]),
]

test_pipeline = val_pipeline

data = dict(
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline),
)
