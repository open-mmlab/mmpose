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

### 2d Face Keypoint Estimation

#### Results on AFLW dataset

The model is trained on AFLW train and evaluated on AFLW full and frontal.

| Arch  | Input Size | NME<sub>*full*</sub> | NME<sub>*frontal*</sub>  | ckpt | log |
| :-------------- | :-----------: | :------: | :------: |:------: |:------: |
| [dark_pose_hrnetv2_w18](/configs/face/darkpose/aflw/hrnetv2_w18_aflw_256x256_dark.py)  | 256x256 | 1.41 | 1.27 | [ckpt](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_aflw_256x256_dark-219606c0_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_aflw_256x256_dark_20210125.log.json) |

#### Results on WFLW dataset

The model is trained on WFLW train.

| Arch  | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> | ckpt | log |
| :-----| :--------: | :------------------: | :------------------: |:---------------------------: |:------------------------: | :------------------: | :--------------: |:-------------------------: |:---: | :---: |
| [dark_pose_hrnetv2_w18](/configs/face/darkpose/wflw/hrnetv2_w18_wflw_256x256_dark.py)  | 256x256 | 3.98  | 6.99 | 3.96  | 4.78  | 4.57  | 3.87  | 4.30  | [ckpt](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_wflw_256x256_dark-3f8e0c2c_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_wflw_256x256_dark_20210125.log.json) |
