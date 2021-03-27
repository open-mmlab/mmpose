# Distribution-aware coordinate representation for human pose estimation

## Introduction

[ALGORITHM]

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

#### Results on RHD test set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18_dark](/configs/hand/darkpose/rhd2d/hrnetv2_w18_rhd2d_256x256_dark.py) | 256x256 | 0.988 | 0.896 | 2.38 | [ckpt](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark-378d761d_20210327.pth) | [log](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark_20210327.log.json) |
