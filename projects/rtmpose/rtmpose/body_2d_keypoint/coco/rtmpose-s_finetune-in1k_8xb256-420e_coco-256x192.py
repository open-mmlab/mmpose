_base_ = ['./rtmpose-s_8xb256-420e_coco-256x192.py']

_base_['model']['backbone']['init_cfg'][
    'checkpoint'] = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth'  # noqa
