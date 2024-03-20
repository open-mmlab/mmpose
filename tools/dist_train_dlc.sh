#!/usr/bin/env bash
cd /mnt/data/mmperc/lupeng/pose/mmpose

pip install openmim
mim install /mnt/data/mmperc/lupeng/repos/mmengine
mim install mmcv
mim install -e /mnt/data/mmperc/lupeng/repos/mmdetection
mim install -e /mnt/data/mmperc/lupeng/repos/mmyolo
mim install -e .
mkdir -p ~/.cache/torch/hub/
ln -s /mnt/data/mmperc/lupeng/checkpoints ~/.cache/torch/hub/

export http_proxy=http://58.34.83.134:31280/
export https_proxy=http://58.34.83.134:31280/

CONFIG=$1
WORK_DIR=${CONFIG/configs/logs}
WORK_DIR=${WORK_DIR%.py}
GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --launcher pytorch \
    --amp \
    --work-dir $WORK_DIR ${@:2}
