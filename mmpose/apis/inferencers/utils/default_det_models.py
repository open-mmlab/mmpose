# Copyright (c) OpenMMLab. All rights reserved.
default_det_models = dict(
    human=dict(model='rtmdet-s', weights=None, cat_ids=(0, )),
    face=dict(
        model='demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py',
        weights='https://download.openmmlab.com/mmpose/mmdet_pretrained/'
        'yolo-x_8xb8-300e_coco-face_13274d7c.pth',
        cat_ids=(0, )),
    hand=dict(
        model='demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py',
        weights='https://download.openmmlab.com/mmpose/mmdet_pretrained/'
        'cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth',
        cat_ids=(0, )),
    animal=dict(
        model='rtmdet-s',
        weights=None,
        cat_ids=(15, 16, 17, 18, 19, 20, 21, 22, 23)),
)

default_det_models['body'] = default_det_models['human']
default_det_models['wholebody'] = default_det_models['human']
