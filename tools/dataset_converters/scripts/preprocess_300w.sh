#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/300w/raw/300w.tar.gz.00 -C $DOWNLOAD_DIR/
tar -xvf $DOWNLOAD_DIR/300w/300w.tar.00 -C $DATA_ROOT/
rm -rf $DOWNLOAD_DIR/300w
