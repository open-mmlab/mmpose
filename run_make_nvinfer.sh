#!/bin/bash

MODEL_DIR=$1
MODEL_DIR=${MODEL_DIR%%/}
shift

OPERATED_ON_CLASS_NAME=$1
shift

CLASSES=( "$@" )
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")

echo "[property]

# model loading.
onnx-file=keypoint_detector.onnx
model-engine-file=keypoint_detector.onnx_b8_gpu0_fp16.engine

# model config
infer-dims=3;256;192
gie-unique-id=2
operate-on-class-ids=0
operated-on-class-name=$OPERATED_ON_CLASS_NAME

[custom]
kp-names=$CLASSES
" > "$MODEL_DIR/keypoints-config.txt"