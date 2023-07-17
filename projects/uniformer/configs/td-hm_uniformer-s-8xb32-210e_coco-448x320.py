_base_ = ['./_base_/td-hm_uniformer-b-8xb32-210e_coco-448x320.py']

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', interval=5))

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1.0e-3,
))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(320, 488), heatmap_size=(80, 112), sigma=3)

model = dict(
    backbone=dict(
        depths=[3, 4, 8, 3],
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='$PATH_TO_YOUR_uniformer_small_in1k.pth')),
    test_cfg=dict())

train_dataloader = dict(batch_size=32)
val_dataloader = dict(batch_size=256)
