#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/MPII_Human_Pose/raw/MPII_Human_Pose.tar.gz -C $DATA_ROOT
rm -rf $DOWNLOAD_DIR/MPII_Human_Pose
