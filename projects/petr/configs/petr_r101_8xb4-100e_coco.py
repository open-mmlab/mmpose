_base_ = ['petr_r50_8xb4-100e_coco.py']

# model
checkpoint = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/' \
    'deformable_detr_twostage_refine_r101_16x2_50e_coco-3186d66b_20230613.pth'

model = dict(init_cfg=dict(checkpoint=checkpoint), backbone=dict(depth=101))
