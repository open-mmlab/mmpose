_base_ = ['./rtmpose-l_8xb32-270e_coco-wholebody-384x288.py']

_base_['model']['backbone']['init_cfg'][
    'checkpoint'] = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'  # noqa
