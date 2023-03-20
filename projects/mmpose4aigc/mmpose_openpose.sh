#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.

python openpose_visualization.py \
    --det_config ../rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    --det_checkpoint models/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
