#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/AP-10K/raw/AP-10K.tar.gz.00 -C $DATA_ROOT
rm -rf $DATA_ROOT/AP-10K
