# InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@InProceedings{Moon_2020_ECCV_InterHand2.6M,
author = {Moon, Gyeongsik and Yu, Shoou-I and Wen, He and Shiratori, Takaaki and Lee, Kyoung Mu},
title = {InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020}
}
```

## Results and models

### 3d Hand Pose Estimation

#### Results on InterHand2.6M val & test set

|Train Set| Set | Arch  | Input Size | MPJPE-single |  MPJPE-interacting  |  MPJPE-all  | MRRPE | APh   | ckpt    | log     |
| :--- | :--- | :--------: | :--------: | :------: | :------: | :------: |:------: |:------: |:------: |:------: |
| All | test(H+M) | [InterNet_resnet_50](/configs/hand/3d_kpt_sview_rgb_img/InterNet/interhand3d/res50_interhand3d_all_256x256.py) | 256x256 | 10.16 | 15.27 | 12.97 | 33.14 | 0.99 | [ckpt](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth) | [log](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256_20210506.log.json) |
| All | val(M) | [InterNet_resnet_50](/configs/hand/3d_kpt_sview_rgb_img/InterNet/interhand3d/res50_interhand3d_all_256x256.py) | 256x256 | 12.03 | 17.88 | 14.84 | 34.93 | 0.99 | [ckpt](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth) | [log](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256_20210506.log.json) |
