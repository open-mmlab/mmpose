#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

unzip $DOWNLOAD_DIR/COCO_2017/raw/Images/val2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/COCO_2017/raw/Images/train2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/COCO_2017/raw/Annotations/annotations_trainval2017.zip -d $DATA_ROOT
rm -rf $DOWNLOAD_DIR/COCO_2017
