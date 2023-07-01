_base_ = ['./_base_/td-hm_uniformer-b-8xb32-210e_coco-448x320.py']

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater', interval=5))

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1.0e-3,
))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(320, 488), heatmap_size=(80, 112), sigma=3)

model = dict(
    # pretrained='/path/to/hrt_small.pth', # Set the path to pretrained backbone here
    backbone=dict(
        layers=[3, 4, 8, 3],
        drop_path_rate=0.2),
    test_cfg=dict())

train_dataloader = dict(batch_size=32)
val_dataloader = dict(batch_size=256)
