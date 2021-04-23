# Deeppose: Human pose estimation via deep neural networks

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

## Results and models

### 2d Hand Keypoint Estimation

#### Results on OneHand10K val set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [deeppose_resnet_50](/configs/hand/deeppose/onehand10k/deeppose_res50_onehand10k_256x256.py) | 256x256 | 0.990 | 0.486 | 34.28 | [ckpt](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_onehand10k_256x256-cbddf43a_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_onehand10k_256x256_20210330.log.json) |

#### Results on CMU Panoptic (MPII+NZSL val set)

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [deeppose_resnet_50](/configs/hand/deeppose/panoptic/deeppose_res50_panoptic_256x256.py) | 256x256 | 0.999 | 0.686 | 9.36 | [ckpt](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_panoptic_256x256-8a745183_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_panoptic_256x256_20210330.log.json) |

#### Results on RHD test set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [deeppose_resnet_50](/configs/hand/deeppose/rhd2d/deeppose_res50_rhd2d_256x256.py) | 256x256 | 0.988 | 0.865 | 3.29 | [ckpt](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_rhd2d_256x256-37f1c4d3_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_rhd2d_256x256_20210330.log.json) |
