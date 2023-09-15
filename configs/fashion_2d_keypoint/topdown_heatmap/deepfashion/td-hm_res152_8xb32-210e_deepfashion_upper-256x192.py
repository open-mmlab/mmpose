_base_ = './td-hm_res50_8xb64-210e_deepfashion_upper-256x192.py'

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

model = dict(
    backbone=dict(
        type='ResNet',
        depth=152,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet152')))

train_dataloader = dict(batch_size=32)
