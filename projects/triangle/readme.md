# triangle

## datasets

```
cd path/to/mmpose
mkdir data & cd data
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/triangle_dataset/Triangle_140_Keypoint_Dataset.zip
unzip Triangle_140_Keypoint_Dataset.zip
```

## train

```shell
python tools/train.py projects/triangle/configs/rtmpose-s.py
```
