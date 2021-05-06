# Deeppose + Wingloss

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

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{feng2018wing,
  title={Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks},
  author={Feng, Zhen-Hua and Kittler, Josef and Awais, Muhammad and Huber, Patrik and Wu, Xiao-Jun},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
  year={2018},
  pages ={2235-2245},
  organization={IEEE}
}
```

## Results and models

### 2d Face Keypoint Estimation

#### Results on WFLW dataset

The model is trained on WFLW train.

| Arch  | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> | ckpt | log |
| :-----| :--------: | :------------------: | :------------------: |:---------------------------: |:------------------------: | :------------------: | :--------------: |:-------------------------: |:---: | :---: |
| [deeppose_res50](/configs/face/deeppose/wflw/deeppose_res50_wflw_256x256.py)  | 256x256 | 4.85 | 8.50 | 4.81 | 5.69 | 5.45 | 4.82 | 5.20 | [ckpt](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256-92d0ba7f_20210303.pth) | [log](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_20210303.log.json) |
| [deeppose_res50_wingloss](/configs/face/deeppose/wflw/deeppose_res50_wflw_256x256_wingloss.py)  | 256x256 | 4.64 | 8.25 | 4.59 | 5.56 | 5.26 | 4.59 | 5.07 | [ckpt](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_wingloss-f82a5e53_20210303.pth) | [log](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_wingloss_20210303.log.json) |
