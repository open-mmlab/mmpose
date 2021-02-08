# Simple baselines for human pose estimation and tracking

## Introduction

[BACKBONE]

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

## Results and models

### 2d Hand Pose Estimation

#### Results on OneHand10K val set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenet_v2](/configs/hand/mobilenet_v2/onehand10k/mobilenetv2_onehand10k_256x256.py) | 256x256 | 0.984 | 0.526 | 29.52 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_onehand10k_256x256-55d34d7d_20201218.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_onehand10k_256x256_20201218.log.json) |

#### Results on CMU Panoptic (MPII+NZSL val set)

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenet_v2](/configs/hand/mobilenet_v2/panoptic/mobilenetv2_panoptic_256x256.py) | 256x256 | 0.998 | 0.684 | 10.09 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_panoptic_256x256-b9ec9b68_20201218.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_panoptic_256x256_20201218.log.json) |
