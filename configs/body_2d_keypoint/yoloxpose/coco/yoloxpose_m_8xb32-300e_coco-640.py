_base_ = './yoloxpose_s_8xb32-300e_coco-640.py'

widen_factor = 0.75
deepen_factor = 0.67
checkpoint = 'https://download.openmmlab.com/mmpose/v1/pretrained_models/' \
             'yolox_m_8x8_300e_coco_20230829.pth'

# model settings
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint),
    ),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    head=dict(head_module_cfg=dict(widen_factor=widen_factor)))
