_base_ = ['petr_r50_8xb4-100e_coco.py']

# model

checkpoint = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/' \
    'deformable_detr_twostage_refine_swin_16x1_50e_coco-95953bd1_20230613.pth'

model = dict(
    init_cfg=dict(checkpoint=checkpoint),
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.5,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True),
    neck=dict(in_channels=[384, 768, 1536]))

auto_scale_lr = dict(base_batch_size=16)
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(optimizer=dict(lr=0.0001))
