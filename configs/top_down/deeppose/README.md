# Deeppose: Human pose estimation via deep neural networks

## Introduction

[ALGORITHM]

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [deeppose_resnet_50](/configs/top_down/deeppose/coco/deeppose_res50_coco_256x192.py)  | 256x192 | 0.526 | 0.816 | 0.586 | 0.638 | 0.887 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192-f6de6c0e_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_20210205.log.json) |
| [deeppose_resnet_101](/configs/top_down/deeppose/coco/deeppose_res101_coco_256x192.py) | 256x192 | 0.560 | 0.832 | 0.628 | 0.668 | 0.900 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192-2f247111_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192_20210205.log.json) |
| [deeppose_resnet_152](/configs/top_down/deeppose/coco/deeppose_res152_coco_256x192.py) | 256x192 | 0.583 | 0.843 | 0.659 | 0.686 | 0.907 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192-7df89a88_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192_20210205.log.json) |

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [deeppose_resnet_50](/configs/top_down/deeppose/mpii/deeppose_res50_mpii_256x256.py) | 256x256 | 0.825 | 0.205 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256-c63cd0b6_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_20210203.log.json) |
| [deeppose_resnet_101](/configs/top_down/deeppose/mpii/deeppose_res101_mpii_256x256.py) | 256x256 | 0.840 | 0.222 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_mpii_256x256-87516a90_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_mpii_256x256_20210205.log.json) |
| [deeppose_resnet_152](/configs/top_down/deeppose/mpii/deeppose_res152_mpii_256x256.py) | 256x256 | 0.848 | 0.232 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_mpii_256x256-15f5e6f9_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_mpii_256x256_20210205.log.json) |

