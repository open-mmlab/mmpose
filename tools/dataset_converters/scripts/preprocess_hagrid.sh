#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

cat $DOWNLOAD_DIR/OpenDataLab___HaGRID/raw/*.tar.gz.*  | tar -xvz -C $DATA_ROOT/..
tar -xvf $DATA_ROOT/HaGRID.tar -C $DATA_ROOT/..
rm -rf $DOWNLOAD_DIR/OpenDataLab___HaGRID
