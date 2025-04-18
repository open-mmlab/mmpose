#!/bin/bash

MODEL_DIR=""
OPERATE_ON_CLASS_NAMES=()
CLASSES=()
RES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --operate-on-class-names)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                OPERATE_ON_CLASS_NAMES+=("$1")
                shift
            done
            ;;
        --classes)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CLASSES+=("$1")
                shift
            done
            ;;
        --res)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                RES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

MODEL_DIR=${MODEL_DIR%%/}
OPERATE_ON_CLASS_NAMES=$(IFS=';' ; echo "${OPERATE_ON_CLASS_NAMES[*]}")
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")
RES=$(IFS=';' ; echo "${RES[*]}")

echo "[property]

# model loading.
onnx-file=keypoint_detector.onnx

# model config
infer-dims=3;256;192

[custom]
min-kp-score=0.0
res=$RES
operate-on-class-names=$OPERATE_ON_CLASS_NAMES
kp-names=$CLASSES
" > "$MODEL_DIR/keypoints-config.txt"
