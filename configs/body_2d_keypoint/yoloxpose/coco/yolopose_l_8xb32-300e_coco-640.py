_base_ = './yolopose_s_8xb32-300e_coco-640.py'

widen_factor = 1
deepen_factor = 1
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox' \
    '_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'

# model settings
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint),
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
