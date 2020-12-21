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
| [pose_mobilenet_v2](/configs/hand/mobilenet_v2/onehand10k/mobilenetv2_onehand10k_256x256.py) | 256x256 | 0.984 | 0.526 | 29.52 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_onehand10k_256x256-55d34d7d_20201218.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_onehand10k_256x256_20201218.log.json) |

#### Results on CMU Panoptic (MPII+NZSL val set).

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenet_v2](/configs/hand/mobilenet_v2/panoptic/mobilenetv2_panoptic_256x256.py) | 256x256 | 0.998 | 0.684 | 10.09 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_panoptic_256x256-b9ec9b68_20201218.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_panoptic_256x256_20201218.log.json) |
