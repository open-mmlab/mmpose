# Simple baselines for human pose estimation and tracking

## Introduction
```
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

## Results and models

### 2d Hand Pose Estimation

#### Results on OneHand10K val set.

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/hand/resnet/onehand10k/res50_onehand10k_256x256.py) | 256x256 | 0.985 | 0.536 | 27.3 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256_20200813.log.json) |

#### Results on FreiHand val set.

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/hand/resnet/freihand/res50_freihand_224x224.py) | 224x224 | 0.992 | 0.868 | 3.25 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_freihand_224x224-ff0799bc_20200914.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_freihand_224x224_20200914.log.json) |

#### Results on CMU Panoptic (MPII+NZSL val set).

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/hand/resnet/panoptic/res50_panoptic_256x256.py) | 256x256 | 0.998 | 0.708 | 9.24 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_panoptic_256x256-5f55ca1a_20200925.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_panoptic_256x256_20200925.log.json) |
