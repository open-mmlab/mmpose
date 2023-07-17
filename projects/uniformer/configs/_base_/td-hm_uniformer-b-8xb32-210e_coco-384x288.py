_base_ = ['./td-hm_uniformer-b-8xb32-210e_coco-256x192.py']

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=2e-3,
))

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

model = dict(
    backbone=dict(
        drop_path_rate=0.4,
        init_cfg=dict(type='Pretrained', checkpoint='${PATH_TO_YOUR_uniformer_base_in1k.pth}')),
    test_cfg=dict())

train_dataloader = dict(batch_size=32)
val_dataloader = dict(batch_size=256)
