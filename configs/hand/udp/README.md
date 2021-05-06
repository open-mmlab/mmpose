# The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@InProceedings{Huang_2020_CVPR,
author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

Note that, UDP also adopts the unbiased encoding/decoding algorithm of [DARK](/configs/hand/darkpose/README.md).

## Results and models

### 2d Hand Keypoint Estimation

#### Results on OneHand10K val set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18_udp](/configs/hand/udp/onehand10k/hrnetv2_w18_onehand10k_256x256_udp.py) | 256x256 | 0.990 | 0.572 | 23.87 | [ckpt](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_onehand10k_256x256_udp-0d1b515d_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_onehand10k_256x256_udp_20210330.log.json) |

#### Results on CMU Panoptic (MPII+NZSL val set)

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18_udp](/configs/hand/udp/panoptic/hrnetv2_w18_panoptic_256x256_udp.py) | 256x256 | 0.998 | 0.742 | 7.84 | [ckpt](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_panoptic_256x256_udp-f9e15948_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_panoptic_256x256_udp_20210330.log.json) |

#### Results on RHD test set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18_udp](/configs/hand/udp/rhd2d/hrnetv2_w18_rhd2d_256x256_udp.py) | 256x256 | 0.992 | 0.902 | 2.21 | [ckpt](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_rhd2d_256x256_udp-63ba6007_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_rhd2d_256x256_udp_20210330.log.json) |
