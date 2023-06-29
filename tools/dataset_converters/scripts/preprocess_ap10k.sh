#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/AP-10K/raw/AP-10K.tar.gz.00 -C $DOWNLOAD_DIR/
tar -xvf $DOWNLOAD_DIR/AP-10K/AP-10K.tar.00 -C $DATA_ROOT/
rm -rf $DOWNLOAD_DIR/AP-10K
