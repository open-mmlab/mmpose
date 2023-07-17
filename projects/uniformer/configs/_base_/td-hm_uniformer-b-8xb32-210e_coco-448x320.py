_base_ = ['./td-hm_uniformer-b-8xb32-210e_coco-256x192.py']

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', interval=10))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(320, 488), heatmap_size=(80, 112), sigma=3)

model = dict(
    backbone=dict(
        drop_path_rate=0.55,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='$PATH_TO_YOUR_uniformer_base_in1k.pth')),
    test_cfg=dict())

train_dataloader = dict(batch_size=32)
val_dataloader = dict(batch_size=256)
