#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/LaPa/raw/LaPa.tar.gz -C $DATA_ROOT
rm -rf $DOWNLOAD_DIR/LaPa
