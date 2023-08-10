#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/OpenDataLab___OneHand10K/raw/OneHand10K.tar.gz.00 -C $DOWNLOAD_DIR/
tar -xvf $DOWNLOAD_DIR/OneHand10K/OneHand10K.tar.00 -C $DATA_ROOT/
rm -rf $DOWNLOAD_DIR/OneHand10K $DOWNLOAD_DIR/OpenDataLab___OneHand10K
