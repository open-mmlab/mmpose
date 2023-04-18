_base_ = ['./petr_r50_8xb4-100e_coco.py']

model = dict(
    bbox_head=dict(
        loss_cls=dict(loss_weight=2.0),
        loss_reg=dict(type='L1Loss', loss_weight=80.0),
        loss_reg_aux=dict(type='L1Loss', loss_weight=70.0),
        loss_oks=dict(type='OksLoss', loss_weight=30.0),
        loss_oks_aux=dict(type='OksLoss', loss_weight=20.0),
        loss_hm=dict(type='mmpose.FocalHeatmapLoss', loss_weight=4.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='KptL1Cost', weight=70.0),
                dict(
                    type='OksCost',
                    metainfo='configs/_base_/datasets/coco.py',
                    weight=70.0)
            ])))


train_dataloader = dict(batch_size=8)