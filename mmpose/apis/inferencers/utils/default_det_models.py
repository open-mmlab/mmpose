# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmengine.config.utils import MODULE2PACKAGE
from mmengine.utils import get_installed_path

mmpose_path = get_installed_path(MODULE2PACKAGE['mmpose'])

default_det_models = dict(
    human=dict(model='rtmdet-m', weights=None, cat_ids=(0, )),
    face=dict(
        model=osp.join(mmpose_path, '.mim',
                       'demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py'),
        weights='https://download.openmmlab.com/mmpose/mmdet_pretrained/'
        'yolo-x_8xb8-300e_coco-face_13274d7c.pth',
        cat_ids=(0, )),
    hand=dict(
        model=osp.join(
            mmpose_path, '.mim', 'demo/mmdetection_cfg/'
            'ssdlite_mobilenetv2_scratch_600e_onehand.py'),
        weights='https://download.openmmlab.com/mmpose/mmdet_pretrained/'
        'ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth',
        cat_ids=(0, )),
    animal=dict(
        model='rtmdet-m',
        weights=None,
        cat_ids=(15, 16, 17, 18, 19, 20, 21, 22, 23)),
)

default_det_models['body'] = default_det_models['human']
default_det_models['wholebody'] = default_det_models['human']
