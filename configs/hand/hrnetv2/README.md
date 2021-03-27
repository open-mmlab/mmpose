# Deep high-resolution representation learning for visual recognition

## Introduction

[ALGORITHM]

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
```

## Results and models

### 2d Hand Keypoint Estimation

#### Results on AFLW dataset

#### Results on RHD test set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18](/configs/hand/hrnetv2/rhd2d/hrnetv2_w18_rhd2d_256x256.py) | 256x256 | 0.987 | 0.895 | 2.41 | [ckpt](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_rhd2d_256x256-763f6799_20210327.pth) | [log](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_rhd2d_256x256_20210327.log.json) |
