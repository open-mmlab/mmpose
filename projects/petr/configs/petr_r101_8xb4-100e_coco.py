_base_ = ['petr_r50_8xb4-100e_coco.py']

# model
model = dict(backbone=dict(depth=101))
