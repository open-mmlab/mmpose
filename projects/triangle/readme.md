# triangle

## datasets

```
cd path/to/mmpose
mkdir data & cd data
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/triangle_dataset/Triangle_Mini_Dataset_Coco.zip
unzip Triangle_Mini_Dataset_Coco.zip
```

## train

```shell
python tools/train.py projects/triangle/configs/rtmpose-s_8xb256-420e_coco-256x192.py
```
