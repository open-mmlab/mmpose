#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/WFLW/raw/WFLW.tar.gz.00 -C $DATA_ROOT
rm -rf $DATA_ROOT/WFLW
