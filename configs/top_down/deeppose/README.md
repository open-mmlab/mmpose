# Deeppose: Human pose estimation via deep neural networks

## Introduction

[ALGORITHM]

```latex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/deeppose/mpii/deeppose_res50_mpii_256x256.py) | 256x256 | 0.825 | 0.205 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256-c63cd0b6_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_20210203.log.json) |
