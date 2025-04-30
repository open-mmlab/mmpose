_base_ = ['./yolox-pose_s_8xb32-300e_coco.py']

# model settings
model = dict(
    init_cfg=dict(checkpoint='https://download.openmmlab.com/mmyolo/v0/yolox/'
                  'yolox_m_fast_8xb32-300e-rtmdet-hyp_coco/yolox_m_fast_8xb32'
                  '-300e-rtmdet-hyp_coco_20230210_144328-e657e182.pth'),
    backbone=dict(
        deepen_factor=0.67,
        widen_factor=0.75,
    ),
    neck=dict(
        deepen_factor=0.67,
        widen_factor=0.75,
    ),
    bbox_head=dict(head_module=dict(widen_factor=0.75)))

train_dataloader = dict(batch_size=64)
