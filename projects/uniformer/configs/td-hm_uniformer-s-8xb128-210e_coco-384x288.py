_base_ = ['./td-hm_uniformer-b-8xb32-210e_coco-384x288.py']

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=2e-3,
))

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

model = dict(
    backbone=dict(
        depths=[3, 4, 8, 3],
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'uniformer/uniformer_small_in1k.pth'  # noqa
        )))

train_dataloader = dict(batch_size=128)
val_dataloader = dict(batch_size=256)
