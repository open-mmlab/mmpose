#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/WFLW/raw/WFLW.tar.gz.00 -C $DOWNLOAD_DIR/
tar -xvf $DOWNLOAD_DIR/WFLW/WFLW.tar.00 -C $DATA_ROOT/
rm -rf $DOWNLOAD_DIR/WFLW
