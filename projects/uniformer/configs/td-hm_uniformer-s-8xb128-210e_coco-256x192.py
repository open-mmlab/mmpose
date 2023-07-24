_base_ = ['./td-hm_uniformer-b-8xb128-210e_coco-256x192.py']

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

train_dataloader = dict(batch_size=32)
val_dataloader = dict(batch_size=256)
