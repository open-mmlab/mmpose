# Aggregated residual transformations for deep neural networks

## Introduction

<!-- [BACKBONE] -->

```bibtex
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnext_50](/configs/top_down/resnext/coco/resnext50_coco_256x192.py)  | 256x192 | 0.714 | 0.898 | 0.789 | 0.771 | 0.937 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_256x192-dcff15f6_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_256x192_20200727.log.json) |
| [pose_resnext_50](/configs/top_down/resnext/coco/resnext50_coco_384x288.py)  | 384x288 | 0.724 | 0.899 | 0.794 | 0.777 | 0.935 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_384x288-412c848f_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_384x288_20200727.log.json) |
| [pose_resnext_101](/configs/top_down/resnext/coco/resnext101_coco_256x192.py) | 256x192 | 0.726 | 0.900 | 0.801 | 0.782 | 0.940 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_256x192-c7eba365_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_256x192_20200727.log.json) |
| [pose_resnext_101](/configs/top_down/resnext/coco/resnext101_coco_384x288.py) | 384x288 | 0.743 | 0.903 | 0.815 | 0.795 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_384x288-f5eabcd6_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_384x288_20200727.log.json) |
| [pose_resnext_152](/configs/top_down/resnext/coco/resnext152_coco_256x192.py) | 256x192 | 0.730 | 0.904 | 0.808 | 0.786 | 0.940 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_256x192-102449aa_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_256x192_20200727.log.json) |
| [pose_resnext_152](/configs/top_down/resnext/coco/resnext152_coco_384x288.py) | 384x288 | 0.742 | 0.902 | 0.810 | 0.794 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_384x288-806176df_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_384x288_20200727.log.json) |

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_resnext_152](/configs/top_down/resnext/mpii/resnext152_mpii_256x256.py) | 256x256 | 0.887 | 0.294 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_mpii_256x256-df302719_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_mpii_256x256_20200927.log.json) |
