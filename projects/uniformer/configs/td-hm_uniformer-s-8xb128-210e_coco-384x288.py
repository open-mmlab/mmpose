_base_ = ['./_base_/td-hm_uniformer-b-8xb32-210e_coco-384x288.py']

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=2e-3,
))

model = dict(
    backbone=dict(
        depths=[3, 4, 8, 3],
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='$PATH_TO_YOUR_uniformer_small_in1k.pth')),
    test_cfg=dict())

train_dataloader = dict(batch_size=128)
val_dataloader = dict(batch_size=256)
