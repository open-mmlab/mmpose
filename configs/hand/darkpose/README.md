# Distribution-aware coordinate representation for human pose estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
}
```

## Results and models

### 2d Hand Keypoint Estimation

#### Results on OneHand10K val set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18_dark](/configs/hand/darkpose/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py) | 256x256 | 0.990 | 0.573 | 23.84 | [ckpt](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark_20210330.log.json) |

#### Results on CMU Panoptic (MPII+NZSL val set)

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18_dark](/configs/hand/darkpose/panoptic/hrnetv2_w18_panoptic_256x256_dark.py) | 256x256 | 0.999 | 0.745 | 7.77 | [ckpt](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_panoptic_256x256_dark-1f1e4b74_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_panoptic_256x256_dark_20210330.log.json) |

#### Results on RHD test set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18_dark](/configs/hand/darkpose/rhd2d/hrnetv2_w18_rhd2d_256x256_dark.py) | 256x256 | 0.992 | 0.903 | 2.17 | [ckpt](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark_20210330.log.json) |
