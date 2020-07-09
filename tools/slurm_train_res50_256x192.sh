#!/usr/bin/env bash

node=Pose
numGPU=8
allGPU=8

CONFIG=configs/top_down/resnet/coco/res50_coco_256x192.py
WORK_DIR='work_dirs/res50_coco_256x192/'

PY_ARGS=${@:5}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

OMPI_MCA_mpi_warn_on_fork=0 \
srun -p $node \
    --gres=gpu:$numGPU \
    -n$allGPU \
    --ntasks-per-node=$numGPU \
    --job-name=$expID \
    --kill-on-bad-exit=1 \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}



srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=8 --job-name=simple_baseline --kill-on-bad-exit=1  python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
