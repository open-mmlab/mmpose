#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.

INPUT_IMAGE=$1

python openpose_visualization.py \
    --det_config ../rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    --det_checkpoint models/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    --pose_config ../rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    --pose_checkpoint models/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input $INPUT_IMAGE \
    --device cpu \
