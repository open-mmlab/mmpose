# Improving Convolutional Networks with Self-Calibrated Convolutions

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{liu2020improving,
  title={Improving Convolutional Networks with Self-Calibrated Convolutions},
  author={Liu, Jiang-Jiang and Hou, Qibin and Cheng, Ming-Ming and Wang, Changhu and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10096--10105},
  year={2020}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_scnet_50](/configs/top_down/scnet/coco/scnet50_coco_256x192.py)   | 256x192 | 0.728 | 0.899 | 0.807 | 0.784 | 0.938 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192-6920f829_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192_20200709.log.json) |
| [pose_scnet_50](/configs/top_down/scnet/coco/scnet50_coco_384x288.py)   | 384x288 | 0.751 | 0.906 | 0.818 | 0.802 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_384x288-9cacd0ea_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_384x288_20200709.log.json) |
| [pose_scnet_101](/configs/top_down/scnet/coco/scnet101_coco_256x192.py)  | 256x192 | 0.733 | 0.903 | 0.813 | 0.790 | 0.941 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_256x192-6d348ef9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_256x192_20200709.log.json) |
| [pose_scnet_101](/configs/top_down/scnet/coco/scnet101_coco_384x288.py)  | 384x288 | 0.752 | 0.906 | 0.823 | 0.804 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_384x288-0b6e631b_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_384x288_20200709.log.json) |

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_scnet_50](/configs/top_down/scnet/mpii/scnet50_mpii_256x256.py) | 256x256 | 0.888 | 0.290 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_mpii_256x256-a54b6af5_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_mpii_256x256_20200812.log.json) |
| [pose_scnet_101](/configs/top_down/scnet/mpii/scnet101_mpii_256x256.py) | 256x256 | 0.886 | 0.293 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_mpii_256x256-b4c2d184_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_mpii_256x256_20200812.log.json) |
