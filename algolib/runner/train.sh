#!/bin/bash
set -x
 
# 0. placeholder
workdir=$(cd $(dirname $1); pwd)
if [[ "$workdir" =~ "submodules/mmpose" ]]
then
    if [ -d "$workdir/algolib/configs" ]
    then
        rm -rf $workdir/algolib/configs
        ln -s $workdir/configs $workdir/algolib/
    else
        ln -s $workdir/configs $workdir/algolib/
    fi
else
    if [ -d "$workdir/submodules/mmpose/algolib/configs" ]
    then
        rm -rf $workdir/submodules/mmpose/algolib/configs
        ln -s $workdir/submodules/mmpose/configs $workdir/submodules/mmpose/algolib/
    else
        ln -s $workdir/submodules/mmpose/configs $workdir/submodules/mmpose/algolib/
    fi
fi
 
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmpose/$3
export PYTORCH_VERSION=1.4
 
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env
path=$PWD
if [[ "$path" =~ "submodules/mmpose" ]]
then
    pyroot=$path
    comroot=$path/../..
    init_path=$path/..
else
    pyroot=$path/submodules/mmpose
    comroot=$path
    init_path=$path/submodules
fi
echo $pyroot
export PYTHONPATH=$comroot:$pyroot:$PYTHONPATH
export FRAME_NAME=mmpose    #customize for each frame
export MODEL_NAME=$3

# init_path
export PYTHONPATH=$init_path/common/sites/:$PYTHONPATH # necessary for init
 
# 4. build necessary parameter
partition=$1 
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
 
# 5. model choice
export PARROTS_DEFAULT_LOGGER=FALSE


case $MODEL_NAME in
    "shufflenetv2_coco_256x192")
        FULL_MODEL="body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_256x192"
        ;;
    "hrnet_w32_coco_256x192_udp")
        FULL_MODEL="body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_udp"
        ;;
    "res50_wflw_256x256")
        FULL_MODEL="face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256"
        ;;
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

port=`expr $RANDOM % 10000 + 20000`

set -x

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --cfg-options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
