#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/OneHand10K/raw/OneHand10K.tar.gz.00 -C $DATA_ROOT
rm -rf $DATA_ROOT/OneHand10K
