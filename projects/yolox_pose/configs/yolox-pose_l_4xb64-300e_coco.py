_base_ = ['./yolox-pose_s_8xb32-300e_coco.py']

# model settings
model = dict(
    init_cfg=dict(checkpoint='https://download.openmmlab.com/mmyolo/v0/yolox/'
                  'yolox_l_fast_8xb8-300e_coco/yolox_l_fast_8xb8-300e_'
                  'coco_20230213_160715-c731eb1c.pth'),
    backbone=dict(
        deepen_factor=1.0,
        widen_factor=1.0,
    ),
    neck=dict(
        deepen_factor=1.0,
        widen_factor=1.0,
    ),
    bbox_head=dict(head_module=dict(widen_factor=1.0)))

train_dataloader = dict(batch_size=64)
