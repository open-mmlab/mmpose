_base_ = 'mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    bbox_head=dict(num_classes=1))

train_dataloader = dict(dataset=dict(metainfo=dict(classes=('person', ))))

val_dataloader = dict(dataset=dict(metainfo=dict(classes=('person', ))))
test_dataloader = val_dataloader
