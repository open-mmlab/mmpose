# Deep high-resolution representation learning for visual recognition

## Introduction

<!-- [ALGORITHM] -->

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

### 2d Face Keypoint Estimation

#### Results on AFLW dataset

The model is trained on AFLW train and evaluated on AFLW full and frontal.

| Arch  | Input Size | NME<sub>*full*</sub> | NME<sub>*frontal*</sub>  | ckpt | log |
| :-------------- | :-----------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18](/configs/face/hrnetv2/aflw/hrnetv2_w18_aflw_256x256.py)  | 256x256 | 1.41 | 1.27 | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256_20210125.log.json) |

#### Results on WFLW dataset

The model is trained on WFLW train.

| Arch  | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> | ckpt | log |
| :-----| :--------: | :------------------: | :------------------: |:---------------------------: |:------------------------: | :------------------: | :--------------: |:-------------------------: |:---: | :---: |
| [pose_hrnetv2_w18](/configs/face/hrnetv2/wflw/hrnetv2_w18_wflw_256x256.py)  | 256x256 | 4.06 | 6.98 | 3.99 | 4.83 | 4.59 | 3.92 | 4.33 | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256-2bf032a6_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256_20210125.log.json) |
