# Mobilenetv2: Inverted residuals and linear bottlenecks

## Introduction

<!-- [BACKBONE] -->

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
| [pose_mobilenet_v2](/configs/hand/mobilenet_v2/onehand10k/mobilenetv2_onehand10k_256x256.py) | 256x256 | 0.986 | 0.537 | 28.60 | [ckpt](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_onehand10k_256x256-f3a3d90e_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_onehand10k_256x256_20210330.log.json) |

#### Results on CMU Panoptic (MPII+NZSL val set)

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenet_v2](/configs/hand/mobilenet_v2/panoptic/mobilenetv2_panoptic_256x256.py) | 256x256 | 0.998 | 0.694 | 9.70 | [ckpt](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_panoptic_256x256-b733d98c_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_panoptic_256x256_20210330.log.json) |

#### Results on RHD test set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenet_v2](/configs/hand/mobilenet_v2/rhd2d/mobilenetv2_rhd2d_256x256.py) | 256x256 | 0.985 | 0.883 | 2.80 | [ckpt](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_rhd2d_256x256-85fa02db_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_rhd2d_256x256_20210330.log.json) |
