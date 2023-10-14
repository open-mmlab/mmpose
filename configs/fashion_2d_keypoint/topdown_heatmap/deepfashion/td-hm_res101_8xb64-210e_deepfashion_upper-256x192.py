_base_ = './td-hm_res50_8xb64-210e_deepfashion_upper-256x192.py'

model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
